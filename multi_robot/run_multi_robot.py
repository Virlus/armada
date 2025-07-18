#!/usr/bin/env python3
"""
Main entry point for multi-robot rollouts with human-in-the-loop failure detection.

Usage:
    python run_multi_robot.py --config-name multi_robot_config
"""

import os
import argparse
import yaml
import hydra
from omegaconf import DictConfig

from multi_robot.nodes.communication_hub import CommunicationHub
from multi_robot.nodes.robot_node import RobotNode
from multi_robot.nodes.teleop_node import TeleopNode

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-robot system parameters')
    parser.add_argument('--config', type=str, default='configs/multi_robot_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--node-type', type=str, choices=['hub', 'robot', 'teleop'],
                        help='Type of node to run')
    parser.add_argument('--id', type=int, default=0,
                        help='ID for robot or teleop node')
    
    return parser.parse_args()

def run_hub(config):
    """Run the communication hub"""
    print(f"Starting communication hub on {config.hub.ip}:{config.hub.port}")
    hub = CommunicationHub(config.hub.ip, config.hub.port)
    hub.run()

def run_robot(config, robot_id):
    """Run a robot node"""
    robot_config = config.robots[robot_id]
    print(f"Starting robot node {robot_id} connecting to hub at {config.hub.ip}:{config.hub.port}")
    
    # 加载机器人特定配置
    with hydra.initialize(config_path='diffusion_policy/diffusion_policy/config'):
        robot_cfg = hydra.compose(config_name=robot_config.config_name)
    
    robot = RobotNode(
        config=robot_cfg,
        robot_id=robot_id,
        socket_ip=config.hub.ip,
        socket_port=config.hub.port
    )
    robot.run()

def run_teleop(config, teleop_id):
    """Run a teleop node"""
    teleop_config = config.teleops[teleop_id]
    print(f"Starting teleop node {teleop_id} connecting to hub at {config.hub.ip}:{config.hub.port}")
    
    teleop = TeleopNode(
        teleop_id=teleop_id,
        socket_ip=config.hub.ip,
        socket_port=config.hub.port,
        listen_freq=teleop_config.listen_freq,
        teleop_device=teleop_config.device,
        num_robot=len(config.robots)
    )
    
    # 保持程序运行
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("Teleop node stopped")

def main():
    args = parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 根据节点类型启动相应组件
    if args.node_type == 'hub':
        run_hub(config)
    elif args.node_type == 'robot':
        run_robot(config, args.id)
    elif args.node_type == 'teleop':
        run_teleop(config, args.id)
    else:
        print("Please specify a valid node type: hub, robot, or teleop")

if __name__ == "__main__":
    main()