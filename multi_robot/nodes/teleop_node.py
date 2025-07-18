import re
import numpy as np
import time
import threading
from scipy.spatial.transform import Rotation as R
import pygame

from multi_robot.communication.socket_client import SocketClient
from multi_robot.utils.message_distillation import parse_message_regex
from multi_robot.utils.keyboard_listener import KeyboardListener
from hardware.my_device.sigma import Sigma7
from hardware.my_device.logitechG29_wheel import Controller

class TeleopNode:
    def __init__(self, teleop_id, socket_ip, socket_port, listen_freq=10, teleop_device="sigma", num_robot=1):
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
        
        # Initialize teleop input devices
        if self.teleop_device == "sigma":
            self.sigma = Sigma7(num_robot=num_robot)
            pygame.init()
            self.controller = Controller(0)
        
        # Initialize keyboard listener for control commands
        self.keyboard_listener = KeyboardListener(teleop_device)
        
        # Start status report thread
        self.inform_thread = threading.Thread(target=self.inform_teleop_state, daemon=True)
        self.inform_thread.start()
    
    def get_separator_pattern(self):
        """Get message separator pattern"""
        separators = [
            "EXECUTE_HUMAN_CHECK",
            "SIGMA",
            "TCP_BEFORE_TAKEOVER"
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
        matches = list(pattern.finditer(combined_msg))
        if not matches:
            return [combined_msg]

        if matches[0].start() > 0:
            parts.append(combined_msg[:matches[0].start()])

        for i, match in enumerate(matches):
            start = match.start()
            next_start = matches[i + 1].start() if i < len(matches) - 1 else len(combined_msg)
            content = combined_msg[start:next_start]
            parts.append(content)
        return parts
    
    def handle_message(self, raw_message):
        """Handle received messages"""
        message_list = self.split_combined_messages(raw_message)
        for message in message_list:
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
            else:
                print(f"Unknown command: {message}")
    
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
            last_p, last_r, _ = self.sigma.get_control(rbt_id)
            last_r = last_r.as_quat(scalar_first=True)
            self.socket.send(f"THROTTLE_SHIFT_POSE_from_{self.teleop_id}_to_{rbt_id}:sigma:{last_p.tolist()},{last_r.tolist()}")
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
    
    def main_human_decide(self, rbt_id, request_type):
        """Main human decision logic"""
        if request_type == "failure":
            print("Press {'S'} for SUCCESS / {'F'} for FAILURE / ")
            print("{'C'} for CONTINUING AGENT policy / {'N'} for needing TELEOP-CTRL / {'P'} for playing back trajectory: ", end='', flush=True)
        elif request_type == "timeout":
            print("Press {'S'} for SUCCESS / {'F'} for FAILURE : ", end='', flush=True)
        
        key = input().strip().upper()
        
        key_actions = {
            'S': self.handle_success,
            'F': self.handle_failure,
            'C': self.handle_continue_policy,
            'N': self.handle_need_teleop,
            'P': self.handle_playback_traj
        }
        
        if request_type == "timeout":
            if key not in ['S', 'F']:
                print(f"Unknown key: {key} for request_type: {request_type}")
                self.teleop_state = "idle"
                return
            key_actions[key](rbt_id, request_type)
        elif request_type == "failure":
            if key not in ['S', 'F', 'C', 'N', 'P']:
                print(f"Unknown key: {key} for request_type: {request_type}")
                self.teleop_state = "idle"
                return
            key_actions[key](rbt_id, request_type)
        else:
            print(f"Unknown request_type: {request_type}")
            self.teleop_state = "idle"
    
    def handle_success(self, rbt_id, request_type):
        """Handle success decision"""
        print("Success!")
        print("Start manually resetting environment, press {'F'} when finished: ", end='', flush=True)
        key = input().strip().upper()
        if key == 'F':
            self.socket.send(f"TELEOP_TAKEOVER_RESULT_SUCCESS_from_robot{rbt_id}")
            self.teleop_state = "idle"
    
    def handle_failure(self, rbt_id, request_type):
        """Handle failure decision"""
        print("Failure!")
        print("Start manually resetting environment, press {'F'} when finished: ", end='', flush=True)
        key = input().strip().upper()
        if key == 'F':
            self.socket.send(f"TELEOP_TAKEOVER_RESULT_FAILURE_from_robot{rbt_id}")
            self.teleop_state = "idle"
    
    def handle_continue_policy(self, rbt_id, request_type):
        """Handle continue policy decision"""
        print("Continue Agent Policy!")
        self.socket.send(f"CONTINUE_POLICY_{rbt_id}")
        self.teleop_state = "idle"
    
    def handle_need_teleop(self, rbt_id, request_type):
        """Handle need teleoperation decision"""
        print("Need Teleop!")
        print("Get ready for teleoperation, press {'S'} to START: ", end='', flush=True)
        key = input().strip().upper()
        if key == 'S':
            print("Teleoperation...")
            print("Press {'C'} to CANCEL, {'T'} to ACCEPT, {'N'} to CONTINUE CURRENT POLICY:", end='', flush=True)
            self.teleop_ctrl_start(rbt_id)
    
    def handle_playback_traj(self, rbt_id, request_type):
        """Handle playback trajectory decision"""
        print("Playback Trajectory!")
        self.socket.send(f"PLAYBACK_TRAJ_{rbt_id}")
        self.main_human_decide(rbt_id, request_type)  # Playback should be a temporal behavior
    
    def teleop_ctrl_start(self, rbt_id):
        """Start teleoperation control"""
        self.socket.send(f"TELEOP_CTRL_START_{rbt_id}")
        self.keyboard_listener.start_keyboard_listener()
        listen_thread = threading.Thread(target=self.run_listen_loop, args=(rbt_id,), daemon=True)
        listen_thread.start()
    
    def run_listen_loop(self, rbt_id):
        """Run listen loop for teleoperation commands"""
        time.sleep(0.5)
        interval = 1.0 / self.listen_freq
        self.stop_event = None
        
        while self.stop_event not in ["cancel", "accept", "continue"]:
            start_time = time.time()
            
            # Check for keyboard control events
            if self.keyboard_listener.accept or self.keyboard_listener.cancel or self.keyboard_listener._continue:
                if self.keyboard_listener.cancel:
                    self.stop_event = "cancel"
                    print("Teleoperation cancelled, robot going home...")
                elif self.keyboard_listener.accept:
                    self.stop_event = "accept"
                    print("Teleoperation accepted.")
                else:
                    self.stop_event = "continue"
                    print("Continue current policy.")
                break
            
            # Send commands based on input device
            if self.teleop_device == "keyboard" and self.keyboard_listener.current_cmd:
                self.socket.send(f"COMMAND_from_{self.teleop_id}_to_{rbt_id}:{self.keyboard_listener.current_cmd}")
            elif self.teleop_device == "sigma":
                diff_p, diff_r, width = self.sigma.get_control(rbt_id)
                diff_r = diff_r.as_quat(scalar_first=True)
                throttle = self.controller.get_throttle()
                self.socket.send(f"COMMAND_from_{self.teleop_id}_to_{rbt_id}:sigma:{diff_p.tolist()},{diff_r.tolist()},{width},{throttle}")
            
            # Maintain loop timing
            elapsed = time.time() - start_time
            time.sleep(max(0, interval - elapsed))
        
        # Clean up after teleoperation
        self.keyboard_listener.stop_keyboard_listener()
        time.sleep(0.5)  # Time needed to restore keyboard settings
        self.tele_ctrl_stop(rbt_id)
    
    def tele_ctrl_stop(self, rbt_id):
        """Stop teleoperation control"""
        msg = f"TELEOP_CTRL_STOP_{rbt_id}_for_{self.stop_event}".encode()
        self.socket.send(msg)
        self.teleop_state = "idle"
    
    def inform_teleop_state(self):
        """Periodically report teleop state"""
        inform_freq = 2  # Hz
        while self.running:
            current_time = time.time()
            if current_time - self.last_query >= 1.0 / inform_freq:
                self.last_query = current_time
                self.socket.send(f"INFORM_TELEOP_STATE_{self.teleop_id}_{self.teleop_state}")
            time.sleep(0.01)