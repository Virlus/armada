import re
import numpy as np
import time
import threading
import argparse
import os
from scipy.spatial.transform import Rotation as R
import pygame

from multi_robot.communication.socket_client import SocketClient
from multi_robot.utils.message_distillation import parse_message_regex
from multi_robot.utils.keyboard_listener import KeyboardListener
from hardware.my_device.sigma import Sigma7
from hardware.my_device.logitechG29_wheel import Controller

#TODO:
#1.increase the ctrl freq
#2.move c(ctn),f(finish),d(waste),h(rewind&teleop) to teleop   ok
#3.rewind ok
#4.throttle

def parse_args():
    parser = argparse.ArgumentParser(description='teleop node parameters')
    parser.add_argument('--teleop_id', type=int, required=False, default=0)

    return parser.parse_args()
class TeleopNode:
    def __init__(self, teleop_id, socket_ip, socket_port, listen_freq=10, teleop_device="sigma", num_robot=1,Ta=8):
        self.Ta = Ta
        self.teleop_id = teleop_id
        self.stop_event = None
        self.listen_freq = listen_freq
        self.socket = SocketClient(socket_ip, socket_port, message_handler=self.handle_message)
        self.socket.start_connection()
        self.lock = threading.Lock()
        self.last_query = time.time() - 1
        self.running = True
        self.teleop_state = "idle"  # busy / idle
        self.teleop_device = teleop_device  # keyboard/sigma
        self.keyboard_listener = KeyboardListener(teleop_device)
        
        # Rewind completion flag
        self.rewind_completed = False
        
        # Initialize teleop input devices
        if self.teleop_device == "sigma":
            self.sigma = Sigma7(num_robot=num_robot)
            pygame.init()
            self.controller = Controller(0)
    
    def get_separator_pattern(self):
        """Get message separator pattern"""
        separators = [
            "EXECUTE_HUMAN_CHECK",
            "SIGMA",
            "TCP_BEFORE_TAKEOVER",
            "REWIND_ROBOT",
            "REWIND_COMPLETED", 
            "SCENE_ALIGNMENT_REQUEST",
            "SCENE_ALIGNMENT_WITH_REF_REQUEST"
        ]
        sorted_seps = sorted(separators, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_seps))
        return re.compile(f"({pattern})")
    
    def split_combined_messages(self, combined_msg):
        """Split combined messages"""
        if not combined_msg:
            return []

        pattern = self.get_separator_pattern()
        parts = []
        last_end = 0
        matches = list(pattern.finditer(combined_msg))
        if not matches:
            return [combined_msg]

        if matches[0].start() > 0:
            parts.append(combined_msg[:matches[0].start()])

        for i, match in enumerate(matches):
            start = match.start()
            end = match.end()
            current_sep = match.group(0)
            next_start = matches[i + 1].start() if i < len(matches) - 1 else len(combined_msg)
            content = combined_msg[start:next_start]
            parts.append(content)
            last_end = next_start
        return parts

    def handle_message(self, raw_message):
        """Handle received messages"""
        message_list = self.split_combined_messages(raw_message)
        for message in message_list:
            print(f"Received message: {message}")
            if message.startswith("EXECUTE_HUMAN_CHECK"):
                human_thread = threading.Thread(target=self.human_decide_process, args=(message,), daemon=True)
                human_thread.start()
            elif message.startswith("SIGMA"):
                assert self.teleop_device == "sigma"
                if "DETACH" in message:
                    self.handle_sigma_detach(message)
                elif "RESUME" in message:
                    self.handle_sigma_resume(message)
                elif "RESET" in message:
                    self.handle_sigma_reset(message)
                elif "TRANSFORM" in message:
                    self.handle_tcp_transform_robot(message)
            elif message.startswith("REWIND_COMPLETED"):
                self.handle_rewind_completed(message)
            elif message.startswith("SCENE_ALIGNMENT_REQUEST"):
                alignment_thread = threading.Thread(target=self.handle_scene_alignment_request, args=(message,), daemon=True)
                alignment_thread.start()
            elif message.startswith("SCENE_ALIGNMENT_WITH_REF_REQUEST"): #rewind
                alignment_thread = threading.Thread(target=self.handle_scene_alignment_with_ref_request, args=(message,), daemon=True)
                alignment_thread.start()
            else:
                raise ValueError(f"Unknown command: {message}")

    def handle_sigma_detach(self, message):
        """Handle sigma detach command"""
        templ = "SIGMA_of_{}_DETACH_from_{}"
        teleop_id, rbt_id = parse_message_regex(message, templ)
        self.sigma.detach(rbt_id)
    
    def handle_sigma_reset(self, message):
        """Handle sigma reset command"""
        templ = "SIGMA_of_{}_RESET_from_{}"
        teleop_id, rbt_id = parse_message_regex(message, templ)
        self.sigma.reset(rbt_id)
    
    def handle_sigma_resume(self, message):
        """Handle sigma resume command"""
        if "DURING_TELEOP" in message:
            templ = "SIGMA_of_{}_RESUME_from_{}_DURING_TELEOP"
            teleop_id, rbt_id = parse_message_regex(message, templ)
            self.sigma.resume(rbt_id)
            print("================================================resumed sigma======================================================")
            last_p, last_r, _ = self.sigma.get_control(rbt_id)
            last_r = last_r.as_quat(scalar_first=True)
            self.socket.send(f"THROTTLE_SHIFT_POSE_from_{self.teleop_id}_to_{rbt_id}:sigma:{last_p.tolist()},{last_r.tolist()}")
            print(f'=================================================sent throttle shift pose: {last_p.tolist()},{last_r.tolist()}============================================')
        else:
            templ = "SIGMA_of_{}_RESUME_from_{}"
            teleop_id, rbt_id = parse_message_regex(message, templ)
            self.sigma.resume(rbt_id)
    
    def handle_tcp_transform_robot(self, msg):
        """Handle TCP transform command"""
        pattern = r"SIGMA_TRANSFORM_from_(\d+)_\[([^\]]+)\]_to_(\d+)"
        match = re.match(pattern, msg)
        if not match:
            raise ValueError(f"Invalid command format: {msg}")
        robot_id = match.group(1)
        tcp_str = match.group(2)
        teleop_id = match.group(3)
        tcp_arr = np.array([float(x.strip()) for x in tcp_str.split(",")])
        assert tcp_arr.shape[0] == 7
        self.sigma.transform_from_robot(
            translate=tcp_arr[0:3],
            rotation=R.from_quat(tcp_arr[3:], scalar_first=True),
            rbt_id=robot_id
        )
    
    def human_decide_process(self, message):
        """
        Human intervention process.
        Key options:
        'S' for Success
        'F' for Failure
        'C' for Continue policy
        'N' for Need teleop-ctrl
        'P' for Playback trajectory
        """
        self.teleop_state = "busy"
        templ = "EXECUTE_HUMAN_CHECK_state_of_robot_{}_with_request{}"
        rbt_id, request_type = parse_message_regex(message, templ)
        print(f"New request from robot {rbt_id}.")
        print(f"Request type: {request_type}")
        
        self.main_human_decide(rbt_id, request_type)
        
        # Ensure teleop_state is properly set after decision process
        # (unless teleop session is starting, which handles its own state)
    
    def main_human_decide(self, rbt_id, request_type):
        """Main human decision logic"""
        self.keyboard_listener.stop_keyboard_listener()  # Stop keyboard listener to prevent conflict with input()
        if request_type == "failure":
            print("Press 'S' for SUCCESS / 'F' for FAILURE / ")
            print("'C' for CONTINUING AGENT policy / 'N' for needing TELEOP-CTRL ", end='', flush=True)
        elif request_type == "timeout":
            print("Press 'S' for SUCCESS / 'F' for FAILURE / ")
            print("'C' for CONTINUING AGENT policy / 'N' for needing TELEOP-CTRL ", end='', flush=True)
        
        key = input().strip().upper()
        
        key_actions = {
            'S': self.handle_success,
            'F': self.handle_failure,
            'C': self.handle_continue_policy,
            'N': self.handle_need_teleop,
            'P': self.handle_playback_traj
        }
        
        if request_type == "timeout":
            if key not in ['S','F','C','N']:
                raise ValueError(f"Unknown key: {key} for request_type: {request_type}")
                if key == 'C':
                    key = 'N'
                    print ("Already timeout, cannot continue policy.")
            key_actions[key](rbt_id,request_type)

        elif request_type == "failure":
            if key not in ['S','F','C','N']:
                raise ValueError(f"Unknown key: {key} for request_type: {request_type}")
            key_actions[key](rbt_id,request_type)

        else:
            raise ValueError(f"Unknown request_type: {request_type}")
        
        # Restart keyboard listener if not entering teleop mode (handle_need_teleop starts its own listener)
        if key != 'N':  # 'N' is for teleop, which starts its own listener
            # Give a moment for any ongoing operations to complete
            time.sleep(0.1)
            # Only restart if teleop_state is idle (not busy with ongoing teleop)
            if self.teleop_state == "idle":
                self.keyboard_listener.start_keyboard_listener()

    def handle_success(self,rbt_id,request_type):
        print("Success!")
        # Note: keyboard_listener already stopped in main_human_decide
        print("Start manually resetting environment , press 'F' when finished: ", end='', flush=True)
        key = input().strip().upper()   # jammed manner,waiting human reset
        if key == 'F':
            msg = f"TELEOP_TAKEOVER_RESULT_SUCCESS_from_robot{rbt_id}".encode()
            self.socket.send(msg)
            self.teleop_state = "idle"

    def handle_failure(self,rbt_id,request_type):
        print("Failure!")
        # Note: keyboard_listener already stopped in main_human_decide
        print("Start manually resetting environment , press 'F' when finished: ", end='', flush=True)
        key = input().strip().upper()  # jammed manner,waiting human reset
        if key == 'F':
            msg = f"TELEOP_TAKEOVER_RESULT_FAILURE_from_robot{rbt_id}".encode()
            self.socket.send(msg)
            self.teleop_state = "idle"

    def handle_continue_policy(self,rbt_id,request_type):
        print("Continue Agent Policy!")
        msg = f"CONTINUE_POLICY_{rbt_id}".encode()
        self.socket.send(msg)
        self.teleop_state = "idle"

    def handle_need_teleop(self,rbt_id,request_type):
        print("Need Teleop!")
        # Note: keyboard_listener already stopped in main_human_decide
        print("Get ready for teleoperation , press 'S' to START: ", end='', flush=True)
        key = input().strip().upper()  # jammed manner,waiting human reset
        if key == 'S':
            print("Rewinding robot before teleoperation...")
            rewind_msg = f"REWIND_ROBOT_{rbt_id}".encode()
            self.socket.send(rewind_msg)
            
            # Wait for rewind completion message
            print("Waiting for robot rewind to complete...")
            self.rewind_completed = False
            while not self.rewind_completed:  #在等一条叫做REWIND_COMPLETED的消息
                time.sleep(0.1)
            time.sleep(1)
            print("Rewind completed. Starting teleoperation...")
            print("Press 'C' or 'c' to CANCEL, 'T' or 't' to ACCEPT, 'N' or 'n' to CONTINUE CURRENT POLICY :", end='', flush=True)
            self.teleop_ctrl_start(rbt_id)  # This will start keyboard_listener for teleop

    def handle_playback_traj(self,rbt_id,request_type):
        print("Playback Trajectory!")
        # Note: keyboard_listener already stopped in main_human_decide
        msg = f"PLAYBACK_TRAJ_{rbt_id}".encode()
        self.socket.send(msg)
        print("Trajectory playback initiated. Please wait and then make another decision.")
        # Return to allow main_human_decide to handle subsequent decisions
        # The playback is handled by robot, we don't need recursive call here

    def run_listen_loop(self,rbt_id):
        ready_to_end_flag = False
        time.sleep(0.5)
        almost_stop_t = -1
        interval = 1.0 / self.listen_freq
        self.stop_event = None
        while True:
            start_time = time.time()

            if almost_stop_t != -1 and time.time() - almost_stop_t >= 1.0:  #give robot node one more second to act
                print ("Stop sending command.")
                break

            # Debug keyboard listener state
            if (self.keyboard_listener.accept or self.keyboard_listener.cancel or self.keyboard_listener._continue) and not ready_to_end_flag:
                # print(f"DEBUG: Keyboard state - accept:{self.keyboard_listener.accept}, cancel:{self.keyboard_listener.cancel}, continue:{self.keyboard_listener._continue}")
                if self.keyboard_listener.cancel:
                    self.stop_event = "cancel"
                    print("Teleoperation cancelled,robot going home...")
                elif self.keyboard_listener.accept:
                    self.stop_event = "accept"
                    print("Teleoperation accepted.")
                else:
                    self.stop_event = "continue"
                    print("Continue current policy.")
                self.send_fake_stop(rbt_id)
                almost_stop_t = time.time()
                ready_to_end_flag = True

            if self.teleop_device == "keyboard" and self.keyboard_listener.current_cmd:
                self.socket.send(f"COMMAND_from_{self.teleop_id}_to_{rbt_id}:{self.keyboard_listener.current_cmd}")  #cmd send from here

            elif self.teleop_device == "sigma":
                diff_p, diff_r, width = self.sigma.get_control(rbt_id)  ##TODO:can already add robot_id
                diff_r = diff_r.as_quat(scalar_first = True)
                # Check throttle pedal state (for teleop pausing)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.keyboard.quit = True
                
                throttle = self.controller.get_throttle()
                # if not throttle < -0.9:
                self.socket.send(f"COMMAND_from_{self.teleop_id}_to_{rbt_id}:sigma:{diff_p.tolist()},{diff_r.tolist()},{width},{throttle}") #send realtime no matter who is ctrlling rbt
                elapsed = time.time() - start_time
                time.sleep(max(0, interval - elapsed))

        self.keyboard_listener.stop_keyboard_listener()
        time.sleep(0.5)   #time is needed to restore keyboard settings
        self.tele_ctrl_stop(rbt_id)


    def teleop_ctrl_start(self,rbt_id):
        msg = f"TELEOP_CTRL_START_{rbt_id}".encode()
        self.socket.send(msg)
        self.keyboard_listener.start_keyboard_listener()
        listen_thread = threading.Thread(target=self.run_listen_loop, args=(rbt_id,),daemon=True)
        listen_thread.start()


    def tele_ctrl_stop(self,rbt_id):
        self.teleop_state = "idle"
        
    def send_fake_stop(self,rbt_id):
        msg = f"TELEOP_CTRL_STOP_{rbt_id}_for_{self.stop_event}".encode()
        self.socket.send(msg)

    def handle_rewind_completed(self, message):
        """Handle rewind completion notification from robot"""
        templ = "REWIND_COMPLETED_{}"
        rbt_id = parse_message_regex(message, templ)[0]
        print(f"Rewind completed for robot {rbt_id}")
        # self.rewind_completed = True

    def handle_scene_alignment_request(self, message):
        """Handle scene alignment request from robot"""
        self.teleop_state = "busy"
        templ = "SCENE_ALIGNMENT_REQUEST_{}_{}"
        rbt_id, context_info = parse_message_regex(message, templ)
        
        print(f"Scene alignment requested for robot {rbt_id}, context: {context_info}")
        print("Robot is displaying alignment images. Please align the scene and press 'C' to continue")
        
        # Stop keyboard listener before using input()
        self.keyboard_listener.stop_keyboard_listener()
        
        # Handle user input for scene alignment confirmation
        key = input().strip().upper()
        while key != 'C':
            print("Press 'C' to continue when scene is aligned: ", end='', flush=True)
            key = input().strip().upper()
        
        # Send completion message
        completion_msg = f"SCENE_ALIGNMENT_COMPLETED_{rbt_id}".encode()
        self.socket.send(completion_msg)
        self.teleop_state = "idle"
        
        # Restart keyboard listener after completing scene alignment
        time.sleep(0.1)
        self.keyboard_listener.start_keyboard_listener()

    def handle_scene_alignment_with_ref_request(self, message):
        """Handle scene alignment with reference request from robot"""
        self.teleop_state = "busy"
         
        # Simple format without image data - robot displays images locally
        templ = "SCENE_ALIGNMENT_WITH_REF_REQUEST_{}_{}"
        rbt_id, context_info = parse_message_regex(message, templ)
        
        print(f"Scene alignment with reference requested for robot {rbt_id}")
        print("Robot is displaying reference alignment images. Please reset the scene and press 'C' to continue")
        
        # Stop keyboard listener before using input()
        self.keyboard_listener.stop_keyboard_listener()
        
        # Handle user input for scene alignment confirmation
        key = input().strip().upper()
        if key == 'C':
            # print("Press 'C' to continue when scene is aligned: ", end='', flush=True)
            # key = input().strip().upper() #todo:here!!!!!!!!!!!!!!!!!!
            self.rewind_completed = True
        
        # Send completion message
        completion_msg = f"SCENE_ALIGNMENT_COMPLETED_{rbt_id}".encode()
        self.socket.send(completion_msg)
        
        # Restart keyboard listener after completing scene alignment
        time.sleep(0.1)
        self.keyboard_listener.start_keyboard_listener()

    def _display_scene_alignment_with_images(self, rbt_id, ref_side_img, ref_wrist_img):
        """Display scene alignment with reference images in teleop node"""
        import cv2
        
        print(f"🖼️  Displaying reference images for robot {rbt_id}")
        print("Please align the scene with the reference images, then press 'C' to continue")
        
        # Stop keyboard listener to prevent conflicts
        self.keyboard_listener.stop_keyboard_listener()
        
        try:
            # Create OpenCV windows
            cv2.namedWindow("Reference_Side", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("Reference_Wrist", cv2.WINDOW_AUTOSIZE) 
            
            # Display reference images
            cv2.imshow("Reference_Side", ref_side_img)
            cv2.imshow("Reference_Wrist", ref_wrist_img)
            cv2.waitKey(1)
            
            print("Reference images displayed. Press 'C' when scene is aligned: ", end='', flush=True)
            
            # Wait for user confirmation
            key = input().strip().upper()
            while key != 'C':
                print("Press 'C' to continue when scene is aligned: ", end='', flush=True)
                key = input().strip().upper()
            
            # Close OpenCV windows
            cv2.destroyAllWindows()
            
            print("✅ Scene alignment confirmed!")
            
        except Exception as e:
            print(f"❌ Error displaying images: {e}")
            # Close windows in case of error
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        finally:
            # Send completion message
            completion_msg = f"SCENE_ALIGNMENT_COMPLETED_{rbt_id}".encode()
            self.socket.send(completion_msg)
            self.teleop_state = "idle"
            
            # Restart keyboard listener
            time.sleep(0.1)
            self.keyboard_listener.start_keyboard_listener()


    def inform_teleop_state(self,inform_freq):
        while self.running:
            if time.time() - self.last_query >= 1/inform_freq:
                # if time.time() - self.last_query >= 1:
                # if self.teleop_device == "sigma":
                    # print("===============================================")
                    # print("init_p_0 = {} ,init_r_0 = {}".format(self.sigma.init_p[0], self.sigma.init_r[0]))
                    # print("init_p_1 = {} ,init_r_1 = {}".format(self.sigma.init_p[1], self.sigma.init_r[1]))
                    # print("prev_p_0 = {} ,prev_r_0 = {}".format(self.sigma._prev_p[0], self.sigma._prev_r[0]))
                    # print("prev_p_1 = {} ,prev_r_1 = {}".format(self.sigma._prev_p[1], self.sigma._prev_r[1]))
                self.last_query = time.time()
                msg = f"INFORM_TELEOP_STATE_{self.teleop_id}_{self.teleop_state}".encode()
                self.socket.send(msg)

if __name__ == "__main__":
    inform_freq = 2
    listen_freq = 30
    teleop_device = "sigma"
    num_robot = 1

    assert teleop_device in ["sigma", "keyboard"]
    args = parse_args()
    args.teleop_id = 0 ##TODO:remember to delete
    # teleop_node = TeleopNode(args.teleop_id,"192.168.1.1", 12345,ctrl_freq,teleop_device,num_robot)
    teleop_node = TeleopNode(args.teleop_id,"127.0.0.1", 12345,listen_freq,teleop_device,num_robot,Ta=8)
    try:
        teleop_state_thread = threading.Thread(    #inform teleop state by a freq
            target=teleop_node.inform_teleop_state,
            args=(inform_freq,),
            daemon = True
        )
        teleop_state_thread.start()

        while True:
            pass

    finally:
        print(1)
        teleop_node.socket.close()



