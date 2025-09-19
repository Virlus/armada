import re
import numpy as np
import time
import threading
import argparse
import os
import sys
from scipy.spatial.transform import Rotation as R
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from armada.communication.socket_client import SocketClient
from armada.utils.message_distillation import parse_message_regex, MessageHandler
from armada.utils.keyboard_listener import KeyboardListener
from hardware.my_device.sigma import Sigma7
from hardware.my_device.logitechG29_wheel import Controller


def parse_args():
    parser = argparse.ArgumentParser(description='teleop node parameters')
    parser.add_argument('--teleop_id', type=int, required=False, default=0)

    return parser.parse_args()


class TeleopNode:
    """Teleoperation node that handles human operator input and intervention.
    Manages teleoperation devices, user interface, and communication with robot nodes."""
    
    def __init__(self, teleop_id, socket_ip, socket_port, listen_freq=10, teleop_device="sigma", num_robot=1, Ta=8):
        # Initialize basic parameters
        self._setup_basic_params(teleop_id, listen_freq, teleop_device, Ta)
        
        # Initialize MessageHandler as a component
        self.message_handler = MessageHandler()
        self._setup_message_routes()
        
        # Setup communication (after message_handler is ready)
        self._setup_communication(socket_ip, socket_port)
        
        # Initialize teleop devices
        self._initialize_devices(num_robot)

    def _setup_basic_params(self, teleop_id, listen_freq, teleop_device, Ta):
        """Setup basic teleop parameters."""
        self.Ta = Ta
        self.teleop_id = teleop_id
        self.stop_event = None
        self.listen_freq = listen_freq
        self.running = True
        self.teleop_state = "idle"  # busy / idle
        self.teleop_device = teleop_device  # keyboard/sigma
        self.keyboard_listener = KeyboardListener(teleop_device)
        self.rewind_completed = False
        self.last_query = time.time() - 1

    def _setup_communication(self, socket_ip, socket_port):
        """Setup communication with hub."""
        self.socket = SocketClient(socket_ip, socket_port, message_handler=self.message_handler.handle_message)
        self.socket.start_connection()
        self.lock = threading.Lock()

    def _initialize_devices(self, num_robot):
        """Initialize teleop input devices."""
        if self.teleop_device == "sigma":
            self.sigma = Sigma7(num_robot=num_robot)
            pygame.init()
            self.controller = Controller(0)

    def _setup_message_routes(self):
        """Define message routing table based on original handle_message logic"""
        
         # Register handlers
        self.threaded_handlers = {
            "EXECUTE_HUMAN_CHECK": self.human_decide_process,
            "SCENE_ALIGNMENT_REQUEST": self.handle_scene_alignment_request,
            "SCENE_ALIGNMENT_WITH_REF_REQUEST": self.handle_scene_alignment_with_ref_request,
        }
        self.unthreaded_handlers = {
            "TIMEOUT":self.handle_rollout_timeout,
        }
        for msg_type, handler in self.threaded_handlers.items():
            self.message_handler.register_handler(msg_type, handler, use_thread=True)
        for msg_type, handler in self.unthreaded_handlers.items():
            self.message_handler.register_handler(msg_type, handler)
            
        self.message_handler.register_pattern_handler(
            lambda msg: msg.startswith("SIGMA"),
            self._handle_sigma_message
        )

    def _handle_sigma_message(self, message: str):
        """Handle SIGMA messages with device assertion and sub-routing"""
        assert self.teleop_device == "sigma"
        
        sigma_handlers = {
            "DETACH": self.handle_sigma_detach,
            "RESUME": self.handle_sigma_resume,
            "RESET": self.handle_sigma_reset,
            "TRANSFORM": self.handle_tcp_transform_robot,
        }
        
        for action, handler in sigma_handlers.items():
            if action in message:
                handler(message)
                break

    def _handle_unknown_message(self, message: str):
        """Custom handler raises exception for unknown messages"""
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
        """Process human decision requests from robot.
        Handles intervention requests and presents options to human operator."""
        
        """
        Human intervention process.
        Key options:
        'S' for Success
        'F' for Failure
        'C' for Continue policy
        'N' for Need teleop-ctrl
        """
        self.teleop_state = "busy"
        templ = "EXECUTE_HUMAN_CHECK_state_of_robot_{}_with_request{}"
        rbt_id, request_type = parse_message_regex(message, templ)
        print(f"===================================New request from robot {rbt_id}.==================================")
        print(f"Request type: {request_type}")
        
        self.main_human_decide(rbt_id, request_type)
        
        # Ensure teleop_state is properly set after decision process
        # (unless teleop session is starting, which handles its own state)
    
    def main_human_decide(self, rbt_id, request_type):
        """Main human decision logic"""
        self.keyboard_listener.stop_keyboard_listener()  # Stop keyboard listener to prevent conflict with input()
        
        # Define valid options and actions
        valid_options = ['S', 'F', 'C', 'N']
        key_actions = {
            'S': self.handle_success,
            'F': self.handle_failure,
            'C': self.handle_continue_policy,
            'N': self.handle_need_teleop,
        }
        if request_type=="timeout": # If no failure signal is raised then consider it as success
            print("Timeout,directly enter next episode....")
            self.handle_success(rbt_id, request_type)
            return
        
        # Display options based on request type
        self._display_decision_options(request_type)
        
        # Get user input
        key = input().strip().upper()
        
        # Handle timeout-specific logic
        if request_type == "timeout" and key == 'C':
            key = 'N'
            print("Already timeout, cannot continue policy, teleop is needed.")
        
        # Validate input
        if key not in valid_options:
            raise ValueError(f"Unknown key: {key} for request_type: {request_type}")
        
        # Execute action
        key_actions[key](rbt_id)

    def _display_decision_options(self, request_type):
        """Display decision options to user."""
        if request_type in ["failure", "timeout"]:
            print("Press 'S' for SUCCESS / 'F' for FAILURE / ")
            print("'C' for CONTINUING AGENT policy / 'N' for needing TELEOP-CTRL ", end='', flush=True)
        else:
            raise ValueError(f"Unknown request_type: {request_type}")

    def handle_success(self,rbt_id):
        """Handle success decision from human operator.
        Processes successful task completion and environment reset."""
        print("Success!")
        msg = f"TELEOP_TAKEOVER_RESULT_SUCCESS_from_robot{rbt_id}".encode()
        self.socket.send(msg)
        self.teleop_state = "idle"

    def handle_failure(self,rbt_id):
        """Handle failure decision from human operator.
        Processes task failure and environment reset."""
        print("Failure!")
        msg = f"TELEOP_TAKEOVER_RESULT_FAILURE_from_robot{rbt_id}".encode()
        self.socket.send(msg)
        self.teleop_state = "idle"

    def handle_continue_policy(self,rbt_id):
        """Handle continue policy decision from human operator.
        Instructs robot to continue autonomous policy execution."""
        print("Continue Agent Policy!")
        msg = f"CONTINUE_POLICY_{rbt_id}".encode()
        self.socket.send(msg)
        self.teleop_state = "idle"

    def handle_need_teleop(self,rbt_id):
        """Handle teleoperation request from human operator.
        Initiates robot rewind and starts teleoperation session."""
        print("Need Teleop!")
        # Note: keyboard_listener already stopped in main_human_decide
        print("Get ready for teleoperation , press 'S' to START: ", end='', flush=True)
        key = input().strip().upper()  # jammed manner,waiting human reset
        if key == 'S':
            print("Rewinding robot before teleoperation...")
            rewind_msg = f"REWIND_ROBOT_{rbt_id}".encode()
            self.socket.send(rewind_msg)
            
            # Wait for rewind completion message
            print("Waiting for robot rewinding to complete...")
            self.rewind_completed = False
            while not self.rewind_completed:  # Wait for REWIND_COMPLETED message
                time.sleep(0.1)
            time.sleep(1)
            print("Rewind completed. Starting teleoperation...")
            print("Press 'C' or 'c' to CANCEL, 'T' or 't' to ACCEPT, 'N' or 'n' to CONTINUE CURRENT POLICY :", end='', flush=True)
            self.teleop_ctrl_start(rbt_id)  # This will start keyboard_listener for teleop

    def run_listen_loop(self,rbt_id):
        """Run listening loop for teleoperation commands.
        Continuously sends commands and monitors for stop conditions."""
        ready_to_end_flag = False
        time.sleep(0.5)
        interval = 1.0 / self.listen_freq
        self.stop_event = None
        while True:
            start_time = time.time()
            last_throttle = False if not "throttle" in locals() else (throttle < -0.9)

            # Check if keyboard listener indicates session should end
            if (self.keyboard_listener.accept or self.keyboard_listener.cancel or self.keyboard_listener._continue) and not ready_to_end_flag:
                if self.keyboard_listener.cancel:
                    self.stop_event = "cancel"
                    print("Teleoperation cancelled,robot going home...")
                elif self.keyboard_listener.accept:
                    self.stop_event = "accept"
                    print("Teleoperation accepted.")
                else:
                    self.stop_event = "continue"
                    print("Continue current policy.")
                self.send_teleop_stop(rbt_id)
                print("Stop sending command.")
                break

            if self.teleop_device == "keyboard" and self.keyboard_listener.current_cmd:
                self.socket.send(f"COMMAND_from_{self.teleop_id}_to_{rbt_id}:{self.keyboard_listener.current_cmd}")  #cmd send from here

            elif self.teleop_device == "sigma":
                curr_time = time.time()
                diff_p, diff_r, width = self.sigma.get_control(rbt_id)
                diff_r = diff_r.as_quat(scalar_first = True)
                # Check throttle pedal state (for teleop pausing)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pass
                
                throttle = self.controller.get_throttle()
                if throttle >= -0.9:
                    if last_throttle:
                        self.sigma.resume(rbt_id)
                        continue
                    self.socket.send(f"COMMAND_from_{self.teleop_id}_to_{rbt_id}:sigma:{diff_p.tolist()},{diff_r.tolist()},{width},{throttle},{curr_time}")
                elif throttle < -0.9 and not last_throttle:
                    self.sigma.detach(rbt_id)
                    self.socket.send(f"COMMAND_from_{self.teleop_id}_to_{rbt_id}:sigma:{diff_p.tolist()},{diff_r.tolist()},{width},{throttle},{curr_time}")
                
                elapsed = time.time() - start_time
                time.sleep(max(0, interval - elapsed))

        self.keyboard_listener.stop_keyboard_listener()
        time.sleep(0.5) # Leave some time for keyboard listener to stop
        self.tele_ctrl_stop()

    def teleop_ctrl_start(self,rbt_id):
        """Start teleoperation control session.
        Sends start command to robot and initializes keyboard listener."""
        msg = f"TELEOP_CTRL_START_{rbt_id}".encode()
        self.socket.send(msg)
        self.keyboard_listener.start_keyboard_listener()
        listen_thread = threading.Thread(target=self.run_listen_loop, args=(rbt_id,),daemon=True)
        listen_thread.start()

    def tele_ctrl_stop(self):
        """Stop teleoperation control session.
        Resets teleop state to idle after session ends."""
        self.teleop_state = "idle"
        
    def send_teleop_stop(self,rbt_id):
        """Send stop command to robot with event type.
        Transmits teleoperation stop signal with appropriate stop event."""
        msg = f"TELEOP_CTRL_STOP_{rbt_id}_for_{self.stop_event}".encode()
        self.socket.send(msg)

    def handle_rollout_timeout(self, message):
        templ = "TIMEOUT_of_{}"
        rbt_id = parse_message_regex(message, templ)[0]
        print(f"=================Robot_{rbt_id} timeout !!!!==============")

    def handle_scene_alignment_request(self, message):
        """Handle scene alignment request from robot.
        Guides human operator through scene alignment process."""
        print(f"===================================New EPISODE STARTED==================================")
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
        # self.keyboard_listener.start_keyboard_listener()

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

        while self.keyboard_listener.listener is not None:
            time.sleep(0.01)

        # Handle user input for scene alignment confirmation
        key = input().strip().upper()
        if key == 'C':
            self.rewind_completed = True
        
        # Send completion message
        completion_msg = f"SCENE_ALIGNMENT_COMPLETED_{rbt_id}".encode()
        self.socket.send(completion_msg)
        
        # Restart keyboard listener after completing scene alignment
        time.sleep(0.1)
        self.keyboard_listener.start_keyboard_listener()

    def inform_teleop_state(self):
        """Report teleop state to hub."""
        msg = f"INFORM_TELEOP_STATE_{self.teleop_id}_{self.teleop_state}".encode()
        self.socket.send(msg)

if __name__ == "__main__":
    listen_freq = 200
    teleop_device = "sigma"
    num_robot = 3

    assert teleop_device in ["sigma", "keyboard"]
    args = parse_args()
    args.teleop_id = 0 
    # for 2 rbts:
    teleop_node = TeleopNode(args.teleop_id,"192.168.1.3", 12345,listen_freq,teleop_device,num_robot,Ta=8)
    # for 1 rbt:
    # teleop_node = TeleopNode(args.teleop_id,"127.0.0.1", 12345,listen_freq,teleop_device,num_robot,Ta=8)
    try:
        teleop_node.inform_teleop_state()

        while True:
            if teleop_node.keyboard_listener.quit:
                completion_msg = f"QUIT".encode()
                teleop_node.socket.send(completion_msg)
                time.sleep(0.5)
                break

    finally:
        teleop_node.socket.close()



