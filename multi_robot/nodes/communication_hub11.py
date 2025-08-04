import numpy as np
import queue
import time
import threading
import re
from typing import Dict, Any, Tuple
from multi_robot.communication.socket_server import SocketServer
from multi_robot.utils.message_distillation import parse_message_regex, split_combined_messages, MessageHandler

class CommunicationHub:
    """Central communication hub that manages message routing between robot and teleop nodes.
    Acts as a broker to handle requests, state updates, and command distribution."""
    
    def __init__(self, socket_ip, socket_port):
        self.running = True
        self.initialize_queue()
        self.lock = threading.Lock()
        
        # Initialize MessageHandler as a component
        self.message_handler = MessageHandler()
        self._setup_message_routes()
        
        self.socket = SocketServer(socket_ip, socket_port, message_handler=self.message_handler.handle_message)
        self.socket.start_connection()
        self.start_scene_alignment_thread()

    def _setup_message_routes(self):
        """Define message routing table for different message types and their handlers.
        Routes are organized by message type with appropriate locks and patterns."""
        
        # Messages that need locks (with self.lock: in original code)
        locked_handlers = {
            "NEED_HUMAN_CHECK": self.cmd_for_add_requestQ,
            "INFORM_TELEOP_STATE": self.update_teleop_queue,
            "INFORM_ROBOT_STATE": self.update_robot_state_dict,
            "TELEOP_TAKEOVER_RESULT": self.report_human_takeover_result,
        }

        # Messages that don't need locks (# with self.lock: commented out in original)
        unlocked_handlers = {
            "CONTINUE_POLICY": self.report_continue_policy,
            "TELEOP_CTRL_START": self.report_teleop_ctrl_start,
            "COMMAND": self.report_teleop_cmd,
            "TELEOP_CTRL_STOP": self.report_teleop_ctrl_stop,
            "THROTTLE_SHIFT": self.report_throttle_shift_pose,
            "REWIND_ROBOT": self.report_rewind_robot,
            "REWIND_COMPLETED": self.report_rewind_completed,
            "SCENE_ALIGNMENT_WITH_REF_REQUEST": self.report_scene_alignment_with_ref_request,
            "SCENE_ALIGNMENT_COMPLETED": self.report_scene_alignment_completed,
            "TIMEOUT":self.handle_demo_timeout
        }
        
        
        # Register handlers
        for msg_type, handler in locked_handlers.items():
            self.message_handler.register_handler(msg_type, handler, use_lock=True, lock_obj=self.lock)
        for msg_type, handler in unlocked_handlers.items():
            self.message_handler.register_handler(msg_type, handler)
        self.message_handler.register_handler("SCENE_ALIGNMENT_REQUEST", self._handle_scene_alignment_request)
        self.message_handler.register_pattern_handler(
            lambda msg: msg.startswith("SIGMA"),
            self._handle_sigma_message
        )

    def _handle_sigma_message(self, message: str, addr):
        """Handle SIGMA device-related messages with appropriate sub-routing.
        Processes different sigma command types (DETACH, RESUME, RESET, TRANSFORM)."""
        sigma_handlers = {
            "DETACH": self.report_sigma_detach,
            "RESUME": self.report_sigma_resume,
            "RESET": self.report_sigma_reset,
            "TRANSFORM": self.report_sigma_transform,
        }
        
        for action, handler in sigma_handlers.items():
            if action in message:
                handler(message, addr)
                break

    def _handle_scene_alignment_request(self, message: str, addr):
        """Special handler for scene alignment requests that places them in a queue.
        Scene alignment requests are processed separately by a dedicated thread."""
        self.scene_alignment_q.put((message, addr))

    def initialize_queue(self):
        """Initialize all communication queues and dictionaries for tracking system state.
        Sets up data structures for robots, teleop nodes, requests, and state tracking."""
        self.robot_dict = {}              # dict  robot_id -> addr
        self.teleop_dict = {}             # dict  teleop_id -> addr
        self.request_q = queue.Queue()    # tuple (robot_id, request_type)
        self.scene_alignment_q = queue.Queue() # tuple (robot_id, context_info)
        self.idle_teleop_q = queue.Queue()    # queue.Queue()  teleop_id
        self.idle_teleop_set = set()      # set to track idle teleop_ids for fast lookup
        self.robot_state_dict = {}        # dict  robot_id -> state
    
    def start_scene_alignment_thread(self):
        """Start a dedicated thread for processing scene alignment requests.
        This allows scene alignment to be handled asynchronously."""
        self.scene_alignment_thread = threading.Thread(target=self.process_scene_alignment_queue, daemon=True)
        self.scene_alignment_thread.start()
        print("Scene alignment processing thread started")
    
    def process_scene_alignment_queue(self):
        """Process scene alignment requests from the queue.
        Pairs alignment requests with available teleop nodes and forwards them."""
        while self.running:
            try:
                if not self.scene_alignment_q.empty() and not self.idle_teleop_q.empty():
                    with self.lock:
                        message, addr = self.scene_alignment_q.get()
                        teleop_id = self.idle_teleop_q.get()  
                        self.idle_teleop_set.discard(teleop_id) 
                        self.report_scene_alignment_request(message, addr)

                time.sleep(0.1)  
            except Exception as e:
                print(f"Error in scene alignment thread: {e}")
                time.sleep(0.1)

    def cmd_for_add_requestQ(self, message, addr):
        """Process human check requests from robots and add them to the request queue.
        Extracts robot ID and request type from the message."""
        rbt_id, request_type = parse_message_regex(message, "NEED_HUMAN_CHECK_from_robot{}_for_{}")
        self.request_q.put((rbt_id, request_type))

    def update_teleop_queue(self, message, addr):
        """Update teleop state information and manage idle teleop queue.
        Tracks which teleop nodes are available for new tasks."""
        teleop_id, teleop_state = parse_message_regex(message, "INFORM_TELEOP_STATE_{}_{}")
        
        if teleop_id not in self.teleop_dict.keys():
            self.teleop_dict[teleop_id] = addr

        if teleop_state == "idle" and teleop_id not in self.idle_teleop_set:
            self.idle_teleop_q.put(teleop_id)
            self.idle_teleop_set.add(teleop_id)
        elif teleop_state == "busy" and teleop_id in self.idle_teleop_set:
            self.idle_teleop_set.remove(teleop_id)
            # Note: teleop_id will be naturally removed from queue when get() is called

    def update_robot_state_dict(self, message, addr):
        """Update robot state information in the central state dictionary.
        Maintains current state of all robots in the system."""
        robot_id, robot_state = parse_message_regex(message, "INFORM_ROBOT_STATE_{}_{}")
        if robot_id not in self.robot_dict.keys():
            self.robot_dict[robot_id] = addr
            self.robot_state_dict[robot_id] = robot_state
        self.robot_state_dict[robot_id] = robot_state

    def report_human_takeover_result(self, message, addr):
        """Forward human takeover results from teleop to robot.
        Handles both success and failure outcomes."""
        templ = "TELEOP_TAKEOVER_RESULT_SUCCESS_from_robot{}" if "SUCCESS" in message else "TELEOP_TAKEOVER_RESULT_FAILURE_from_robot{}"
        rbt_id = parse_message_regex(message, templ)[0]
        send_msg = "TELEOP_TAKEOVER_RESULT_SUCCESS" if "SUCCESS" in message else "TELEOP_TAKEOVER_RESULT_FAILURE"
        self.socket.send(self.robot_dict[rbt_id], send_msg)

    def report_continue_policy(self, message, addr):
        """Forward continue policy command from teleop to robot.
        Instructs robot to continue with autonomous policy execution."""
        templ = "CONTINUE_POLICY_{}"
        rbt_id = parse_message_regex(message, templ)[0]
        send_msg = "CONTINUE_POLICY"
        self.socket.send(self.robot_dict[rbt_id], send_msg)
    
    def handle_demo_timeout(self, message, addr):
        """Forward continue policy command from teleop to robot.
        Instructs robot to continue with autonomous policy execution."""
        self.socket.send(self.teleop_dict["0"], message)


    def report_teleop_ctrl_start(self, message, addr):
        """Forward teleop control start command from teleop to robot.
        Signals robot to prepare for teleoperation mode."""
        templ = "TELEOP_CTRL_START_{}"
        rbt_id = parse_message_regex(message, templ)[0]
        send_msg = "TELEOP_CTRL_START"
        self.socket.send(self.robot_dict[rbt_id], send_msg)

    def report_teleop_ctrl_stop(self, message, addr):
        """Forward teleop control stop command from teleop to robot.
        Signals robot to end teleoperation mode with specified event type."""
        templ = "TELEOP_CTRL_STOP_{}_for_{}"
        rbt_id, stop_event = parse_message_regex(message, templ)
        send_msg = message
        self.socket.send(self.robot_dict[rbt_id], send_msg)

    def report_teleop_cmd(self, message, addr):
        """Forward teleop commands from teleop to robot.
        Handles command parsing and routing to the appropriate robot."""
        messages = message.split('COMMAND_')
        for msg in messages:
            if not msg:
                continue
            full_msg = 'COMMAND_' + msg
            templ = "COMMAND_from_{}_to_{}:{}"
            teleop_id, rbt_id, cmd = parse_message_regex(full_msg, templ)
            send_msg = full_msg
            self.socket.send(self.robot_dict[rbt_id], send_msg)

    def report_sigma_detach(self, message, addr):
        """Forward sigma detach command from robot to teleop.
        Instructs teleop to detach the haptic device from robot control."""
        templ = "SIGMA_of_{}_DETACH_from_{}"
        teleop_id, rbt_id = parse_message_regex(message, templ)
        send_msg = message
        self.socket.send(self.teleop_dict[teleop_id], send_msg)

    def report_sigma_resume(self, message, addr):
        """Forward sigma resume command from robot to teleop.
        Instructs teleop to resume haptic device connection to robot."""
        if "DURING_TELEOP" in message:
            templ = "SIGMA_of_{}_RESUME_from_{}_DURING_TELEOP"
        else:
            templ = "SIGMA_of_{}_RESUME_from_{}"
        teleop_id, rbt_id = parse_message_regex(message, templ)
        send_msg = message
        self.socket.send(self.teleop_dict[teleop_id], send_msg)

    def report_sigma_reset(self, message, addr):
        """Forward sigma reset command from robot to teleop.
        Instructs teleop to reset the haptic device to initial state."""
        templ = "SIGMA_of_{}_RESET_from_{}"
        teleop_id, rbt_id = parse_message_regex(message, templ)
        send_msg = message
        self.socket.send(self.teleop_dict[teleop_id], send_msg)

    def report_sigma_transform(self, message, addr):
        """Forward sigma transform command from robot to teleop.
        Sends transformation data to align haptic device with robot state."""
        templ = "SIGMA_TRANSFORM_from_{}_{}_to_{}"
        rbt_id, _, teleop_id = parse_message_regex(message, templ)
        send_msg = message
        self.socket.send(self.teleop_dict[teleop_id], send_msg)

    def report_throttle_shift_pose(self, message, addr):
        """Forward throttle shift pose information from teleop to robot.
        Transmits pose adjustments based on throttle control input."""
        templ = "THROTTLE_SHIFT_POSE_from_{}_to_{}:{}"
        teleop_id, rbt_id, else_th = parse_message_regex(message, templ)
        send_msg = message
        try:
            self.socket.send(self.robot_dict[rbt_id], send_msg)
        except Exception as e:
            print(f"ERROR sending message to robot {rbt_id}: {e}")

    def report_rewind_robot(self, message, addr):
        """Forward rewind message to robot"""
        templ = "REWIND_ROBOT_{}"
        rbt_id = parse_message_regex(message, templ)[0]
        send_msg = "REWIND_ROBOT"
        self.socket.send(self.robot_dict[rbt_id], send_msg)

    def report_rewind_completed(self, message, addr):
        """Forward rewind completion message to teleop"""
        templ = "REWIND_COMPLETED_{}"
        rbt_id = parse_message_regex(message, templ)[0]
        
        # Find the teleop that initiated the rewind (ideally track this)
        teleop_id = "0"
        if teleop_id is not None:
            send_msg = f"REWIND_COMPLETED_{rbt_id}"
            self.socket.send(self.teleop_dict[teleop_id], send_msg)

    def report_scene_alignment_request(self, message, addr):
        templ = "SCENE_ALIGNMENT_REQUEST_{}_{}"
        rbt_id, context_info = parse_message_regex(message, templ)
        teleop_id = "0"
        send_msg = f"SCENE_ALIGNMENT_REQUEST_{rbt_id}_{context_info}"
        self.socket.send(self.teleop_dict[teleop_id], send_msg)

    def report_scene_alignment_with_ref_request(self, message, addr):  
        templ = "SCENE_ALIGNMENT_WITH_REF_REQUEST_{}_{}"
        teleop_id = "0"
        self.socket.send(self.teleop_dict[teleop_id], message)

    def report_scene_alignment_completed(self, message, addr):
        templ = "SCENE_ALIGNMENT_COMPLETED_{}"
        rbt_id = parse_message_regex(message, templ)[0]
        send_msg = "SCENE_ALIGNMENT_COMPLETED"
        self.socket.send(self.robot_dict[rbt_id], send_msg)

    def update_request_q_workflow(self):
        """Process the top element of the request queue if teleop is available.
        Coordinates human check requests between robots and teleop nodes."""
        if not self.request_q.empty() and not self.idle_teleop_q.empty():
            with self.lock:
                cur_idle_teleop_id = self.idle_teleop_q.get()
                self.idle_teleop_set.remove(cur_idle_teleop_id)
                cur_request = self.request_q.get()

            # Inform the robot
            rbt_id, request_type = cur_request[0], cur_request[1]
            addr_rbt = self.robot_dict[rbt_id]
            msg_rbt = f"READY_for_state_check_by_human_with_teleop_id_{cur_idle_teleop_id}".encode()
            self.socket.send(addr_rbt, msg_rbt)

            # Inform the teleop node
            addr_teleop = self.teleop_dict[cur_idle_teleop_id]
            msg_teleop = f"EXECUTE_HUMAN_CHECK_state_of_robot_{rbt_id}_with_request{request_type}".encode()
            self.socket.send(addr_teleop, msg_teleop)

    def run(self):
        """Main execution loop for the communication hub.
        Continuously processes request queue and handles communication."""
        try:
            while True:
                self.update_request_q_workflow()
                time.sleep(0.8)
        except KeyboardInterrupt:
            self.socket.stop()
            print("Server shutdown")


if __name__ == "__main__":
    server_freq = 50
    #  for 1 or 2 rbts
    hub = CommunicationHub("0.0.0.0", 12345)
    hub.run()

