import numpy as np
import queue
import time
import threading
import re
from multi_robot.communication.socket_server import SocketServer
from multi_robot.utils.message_distillation import parse_message_regex

class CommunicationHub:
    def __init__(self, socket_ip, socket_port):
        self.initialize_queue()
        self.socket = SocketServer(socket_ip, socket_port, message_handler=self.handle_message)
        self.socket.start_connection()
        self.lock = threading.Lock()

    def initialize_queue(self):
        self.robot_dict = {}              # dict  robot_id -> addr
        self.teleop_dict = {}             # dict  teleop_id -> addr
        self.request_q = queue.Queue()    # tuple (robot_id, request_type)
        self.idle_teleop_q = []           # list  teleop_id
        self.robot_state_dict = {}        # dict  robot_id -> state

    def get_separator_pattern(self):
        separators = [
            "NEED_HUMAN_CHECK",
            "INFORM_TELEOP_STATE",
            "INFORM_ROBOT_STATE",
            "TELEOP_TAKEOVER_RESULT",
            "CONTINUE_POLICY",
            "PLAYBACK_TRAJ",
            "TELEOP_CTRL_START",
            "COMMAND",
            "TELEOP_CTRL_STOP",
            "SIGMA",
            "THROTTLE_SHIFT",
            "TCP_BEFORE_TAKEOVER",
            "REWIND_ROBOT",
            "REWIND_COMPLETED",
            "SCENE_ALIGNMENT_REQUEST",
            "SCENE_ALIGNMENT_WITH_REF_REQUEST",
            "SCENE_ALIGNMENT_COMPLETED"
        ]
        sorted_seps = sorted(separators, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_seps))
        return re.compile(f"({pattern})")

    def split_combined_messages(self, combined_msg):
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
            end = match.end()
            current_sep = match.group(0)
            next_start = matches[i + 1].start() if i < len(matches) - 1 else len(combined_msg)
            content = combined_msg[start:next_start]
            parts.append(content)
            last_end = next_start
        return parts

    def handle_message(self, raw_message, addr):
        message_list = self.split_combined_messages(raw_message)
        for message in message_list:
            print("============message:", message)
            if message.startswith("NEED_HUMAN_CHECK"):  # From robot, frequency determined by agent
                with self.lock:
                    self.cmd_for_add_requestQ(message, addr)

            elif message.startswith("INFORM_TELEOP_STATE"):  # From teleop, frequency about 2 Hz
                with self.lock:
                    self.update_teleop_queue(message, addr)

            elif message.startswith("INFORM_ROBOT_STATE"):  # From robot, frequency about 5 Hz
                with self.lock:
                    self.update_robot_state_dict(message, addr)

            elif message.startswith("TELEOP_TAKEOVER_RESULT"):
                with self.lock:
                    self.report_human_takeover_result(message, addr)

            elif message.startswith("CONTINUE_POLICY"):
                # with self.lock:
                self.report_continue_policy(message, addr)

            elif message.startswith("PLAYBACK_TRAJ"):
                # with self.lock:
                self.report_playback_traj(message, addr)

            elif message.startswith("TELEOP_CTRL_START"):
                # with self.lock:
                self.report_teleop_ctrl_start(message, addr)

            elif message.startswith("COMMAND"):
                # with self.lock:
                self.report_teleop_cmd(message, addr)

            elif message.startswith("TELEOP_CTRL_STOP"):
                # with self.lock:
                self.report_teleop_ctrl_stop(message, addr)

            elif message.startswith("SIGMA") and "DETACH" in message:
                # with self.lock:
                self.report_sigma_detach(message, addr)

            elif message.startswith("SIGMA") and "RESUME" in message:
                # with self.lock:
                self.report_sigma_resume(message, addr)

            elif message.startswith("SIGMA") and "RESET" in message:
                # with self.lock:
                self.report_sigma_reset(message, addr)

            elif message.startswith("SIGMA") and "TRANSFORM" in message:
                # with self.lock:
                self.report_sigma_transform(message, addr)

            elif message.startswith("THROTTLE_SHIFT"):
                # with self.lock:
                self.report_throttle_shift_pose(message, addr)

            elif message.startswith("REWIND_ROBOT"):
                # with self.lock:
                self.report_rewind_robot(message, addr)

            elif message.startswith("REWIND_COMPLETED"):
                # with self.lock:
                self.report_rewind_completed(message, addr)

            elif message.startswith("SCENE_ALIGNMENT_REQUEST"):
                # with self.lock:
                self.report_scene_alignment_request(message, addr)

            elif message.startswith("SCENE_ALIGNMENT_WITH_REF_REQUEST"):
                # with self.lock:
                self.report_scene_alignment_with_ref_request(message, addr)

            elif message.startswith("SCENE_ALIGNMENT_COMPLETED"):
                # with self.lock:
                self.report_scene_alignment_completed(message, addr)

            elif message.startswith("Hello"):
                pass
            
            else:
                print(f"Unknown command: {message}")

    def cmd_for_add_requestQ(self, message, addr):
        rbt_id, request_type = parse_message_regex(message, "NEED_HUMAN_CHECK_from_robot{}_for_{}")
        # print("============rbt_id", rbt_id)
        self.request_q.put((rbt_id, request_type))

    def update_teleop_queue(self, message, addr):
        teleop_id, teleop_state = parse_message_regex(message, "INFORM_TELEOP_STATE_{}_{}")
        # print("============teleop_id:{}, teleop_state:{}".format(teleop_id, teleop_state))
        
        if teleop_id not in self.teleop_dict.keys():
            self.teleop_dict[teleop_id] = addr

        if teleop_state == "idle" and teleop_id not in self.idle_teleop_q:
            self.idle_teleop_q.append(teleop_id)
        elif teleop_state == "busy" and teleop_id in self.idle_teleop_q:
            self.idle_teleop_q.remove(teleop_id)

    def update_robot_state_dict(self, message, addr):
        robot_id, robot_state = parse_message_regex(message, "INFORM_ROBOT_STATE_{}_{}")
        print("============robot_id:{}, robot_state:{}".format(robot_id, robot_state))
        if robot_id not in self.robot_dict.keys():
            self.robot_dict[robot_id] = addr
            self.robot_state_dict[robot_id] = robot_state
        self.robot_state_dict[robot_id] = robot_state

    def report_human_takeover_result(self, message, addr):
        templ = "TELEOP_TAKEOVER_RESULT_SUCCESS_from_robot{}" if "SUCCESS" in message else "TELEOP_TAKEOVER_RESULT_FAILURE_from_robot{}"
        rbt_id = parse_message_regex(message, templ)[0]
        # print("============rbt_id:{}".format(rbt_id))
        send_msg = "TELEOP_TAKEOVER_RESULT_SUCCESS" if "SUCCESS" in message else "TELEOP_TAKEOVER_RESULT_FAILURE"
        self.socket.send(self.robot_dict[rbt_id], send_msg)

    def report_continue_policy(self, message, addr):
        templ = "CONTINUE_POLICY_{}"
        rbt_id = parse_message_regex(message, templ)[0]
        print("============rbt_id:{}".format(rbt_id))
        send_msg = "CONTINUE_POLICY"
        self.socket.send(self.robot_dict[rbt_id], send_msg)

    def report_playback_traj(self, message, addr):
        templ = "PLAYBACK_TRAJ_{}"
        rbt_id = parse_message_regex(message, templ)[0]
        # print("============rbt_id:{}".format(rbt_id))
        send_msg = "PLAYBACK_TRAJ"
        self.socket.send(self.robot_dict[rbt_id], send_msg)

    def report_teleop_ctrl_start(self, message, addr):
        templ = "TELEOP_CTRL_START_{}"
        rbt_id = parse_message_regex(message, templ)[0]
        # print("============rbt_id:{}".format(rbt_id))
        send_msg = "TELEOP_CTRL_START"
        self.socket.send(self.robot_dict[rbt_id], send_msg)

    def report_teleop_ctrl_stop(self, message, addr):
        templ = "TELEOP_CTRL_STOP_{}_for_{}"
        rbt_id, stop_event = parse_message_regex(message, templ)
        # print("============rbt_id:{}, stop_event:{}".format(rbt_id, stop_event))   
        send_msg = message
        self.socket.send(self.robot_dict[rbt_id], send_msg)

    def report_teleop_cmd(self, message, addr):
        messages = message.split('COMMAND_')
        for msg in messages:
            if not msg:
                continue
            full_msg = 'COMMAND_' + msg
            templ = "COMMAND_from_{}_to_{}:{}"
            teleop_id, rbt_id, cmd = parse_message_regex(full_msg, templ)
            # print("============teleop_id:{}, rbt_id:{}".format(teleop_id, rbt_id))
            send_msg = full_msg
            self.socket.send(self.robot_dict[rbt_id], send_msg)

    def report_sigma_detach(self, message, addr):
        templ = "SIGMA_of_{}_DETACH_from_{}"
        teleop_id, rbt_id = parse_message_regex(message, templ)
        # print("============teleop_id:{}, rbt_id:{}".format(teleop_id, rbt_id))
        send_msg = message
        self.socket.send(self.teleop_dict[teleop_id], send_msg)

    def report_sigma_resume(self, message, addr):
        if "DURING_TELEOP" in message:
            templ = "SIGMA_of_{}_RESUME_from_{}_DURING_TELEOP"
        else:
            templ = "SIGMA_of_{}_RESUME_from_{}"
        teleop_id, rbt_id = parse_message_regex(message, templ)
        # print("============teleop_id:{}, rbt_id:{}".format(teleop_id, rbt_id)) 
        send_msg = message
        self.socket.send(self.teleop_dict[teleop_id], send_msg)

    def report_sigma_reset(self, message, addr):
        templ = "SIGMA_of_{}_RESET_from_{}"
        teleop_id, rbt_id = parse_message_regex(message, templ)
        # print("============teleop_id:{}, rbt_id:{}".format(teleop_id, rbt_id))  
        send_msg = message
        self.socket.send(self.teleop_dict[teleop_id], send_msg)

    def report_sigma_transform(self, message, addr):
        templ = "SIGMA_TRANSFORM_from_{}_{}_to_{}"
        rbt_id, _, teleop_id = parse_message_regex(message, templ)
        # print("============rbt_id:{}, teleop_id:{}".format(rbt_id, teleop_id))
        send_msg = message
        self.socket.send(self.teleop_dict[teleop_id], send_msg)

    def report_throttle_shift_pose(self, message, addr):
        templ = "THROTTLE_SHIFT_POSE_from_{}_to_{}:{}"
        teleop_id, rbt_id, else_th = parse_message_regex(message, templ)
        print("============teleop_id:{}, rbt_id:{}".format(teleop_id, rbt_id))
        # target_addr = self.robot_dict[rbt_id]
        # conn = self.socket.active_connections.get(target_addr)
        # # 🧪 测试发送一个简单消息
        # test_msg = "COMMAND_CHEAT"
        # try:
        #     print(f"🧪 Sending test message: {test_msg}")
        #     self.socket.send(self.robot_dict[rbt_id], test_msg)
        #     print(f"🧪 Test message sent successfully")
        # except Exception as e:
        #     print(f"🧪 Test message failed: {e}")
        
        # if conn:
        #     # 检查发送缓冲区
        #     try:
        #         import socket
        #         send_buffer_size = conn.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        #         print(f"Send buffer size: {send_buffer_size}")
        #     except:
        #         pass
        send_msg = message
        # send_msg = f"THROTTLE_SHIFT_POSE_from_{teleop_id}_to_{rbt_id}:sigma:{else_th}"
        try:
            self.socket.send(self.robot_dict[rbt_id], send_msg)
            print(f"+++++++++++++++++++++++++++++++++++++++++++robot_dict_id: {self.robot_dict[rbt_id]}")
            print(f"+++++++++++++++++++++++++++++++++++++++++++send_msg: {send_msg}")
            print(f"Message sent successfully to robot {rbt_id}")
        except Exception as e:
            print(f"ERROR sending message to robot {rbt_id}: {e}")
        

    def report_rewind_robot(self, message, addr):
        """Forward rewind message to robot"""
        templ = "REWIND_ROBOT_{}"
        rbt_id = parse_message_regex(message, templ)[0]
        print("============Rewind request for robot_id:{}".format(rbt_id))
        send_msg = "REWIND_ROBOT"
        self.socket.send(self.robot_dict[rbt_id], send_msg)

    def report_rewind_completed(self, message, addr):
        """Forward rewind completion message to teleop"""
        templ = "REWIND_COMPLETED_{}"
        rbt_id = parse_message_regex(message, templ)[0]
        print("============Rewind completed for robot_id:{}".format(rbt_id))
        
        # Find the teleop that initiated the rewind (ideally track this)
        # if len(self.idle_teleop_q) > 0:
            # For simplicity, send to the first available teleop
        teleop_id = "0"
        if teleop_id is not None:
            send_msg = f"REWIND_COMPLETED_{rbt_id}"
            self.socket.send(self.teleop_dict[teleop_id], send_msg)
        # else:
        #     print("No teleop available to receive rewind completion")

    def report_scene_alignment_request(self, message, addr):
        """Forward scene alignment request to teleop"""
        templ = "SCENE_ALIGNMENT_REQUEST_{}_{}"
        rbt_id, context_info = parse_message_regex(message, templ)
        print("============Scene alignment request for robot_id:{}, context:{}".format(rbt_id, context_info))
        
        # Find an idle teleop to handle the request
        # if len(self.idle_teleop_q) > 0:
        teleop_id = "0"
        send_msg = f"SCENE_ALIGNMENT_REQUEST_{rbt_id}_{context_info}"
        self.socket.send(self.teleop_dict[teleop_id], send_msg)
        # else:
            # print("No idle teleop available for scene alignment")

    def report_scene_alignment_with_ref_request(self, message, addr):
        """Forward scene alignment with reference request to teleop"""
        
        # Check if message contains image data
        if "_DATA:" in message:
            # New format with image data - extract robot_id differently
            header_part = message.split("_DATA:")[0]
            try:
                # Try to extract robot_id from header: SCENE_ALIGNMENT_WITH_REF_REQUEST_{robot_id}_rewind
                parts = header_part.split("_")
                if len(parts) >= 4:
                    rbt_id = parts[3]  # robot_id is at index 3
                    context_info = "rewind_with_data"
                else:
                    raise ValueError("Unable to parse robot_id from header")
            except Exception as e:
                print(f"Error parsing message with image data: {e}")
                print(f"Message header: {header_part}")
                return
            
            print(f"============Scene alignment with ref request (with image data) for robot_id:{rbt_id}, context:{context_info}")
            
        else:
            # Old format without image data
            try:
                templ = "SCENE_ALIGNMENT_WITH_REF_REQUEST_{}_{}"
                rbt_id, context_info = parse_message_regex(message, templ)
            except Exception as e:
                print(f"Error parsing old format message: {e}")
                print(f"Message: {message}")
                return
            
            print(f"============Scene alignment with ref request (old format) for robot_id:{rbt_id}, context:{context_info}")
        
        # Find an idle teleop to handle the request
        # if len(self.idle_teleop_q) > 0:
        teleop_id = "0"
        # Forward the entire message as-is to preserve image data
        self.socket.send(self.teleop_dict[teleop_id], message)
        print(f"Forwarded to teleop {teleop_id}")
        # else:
        #     print("No idle teleop available for scene alignment")

    def report_scene_alignment_completed(self, message, addr):
        """Forward scene alignment completion to robot"""
        templ = "SCENE_ALIGNMENT_COMPLETED_{}"
        rbt_id = parse_message_regex(message, templ)[0]
        print("============Scene alignment completed for robot_id:{}".format(rbt_id))
        send_msg = "SCENE_ALIGNMENT_COMPLETED"
        self.socket.send(self.robot_dict[rbt_id], send_msg)

    def update_request_q_workflow(self):
        """Process only the top element of the request queue"""
        if not self.request_q.empty() and len(self.idle_teleop_q) > 0:
            with self.lock:
                cur_idle_teleop_id = self.idle_teleop_q[0]
                self.idle_teleop_q.pop(0)
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
        """Main loop to process the request queue"""
        try:
            while True:
                self.update_request_q_workflow()
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.socket.stop()
            print("Server shutdown")


if __name__ == "__main__":
    server_freq = 50
    #  for 1 or 2 rbts
    hub = CommunicationHub("0.0.0.0", 12345)
    hub.run()

