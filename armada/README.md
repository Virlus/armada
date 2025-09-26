# 🌐 Multi-robot deployment setup

## 🛜 Network Setup via Ethernet

We use a [Unifi Enterprise 8 PoE](https://techspecs.ui.com/unifi/switching/usw-enterprise-8-poe?subcategory=all-switching) as a switch for multi-robot communication.
Each host serves as a node for robot control (robot node) and should be connected to the switch via Ethernet.
Following are the guide for Ethernet connection.

#### Setup connection

1. Connect the host and the switch with Ethernet cable.

2. Check the wired network interface:
```
ip a
```
The name of the second interface should overwrite the `ABC` values below.

3. Configure static IP addresses:
```
sudo gedit /etc/netplan/01-network-manager-all.yaml
```

Override the configuration file with the following content:
```

#---------content-----------
network:
  version: 2
  renderer: networkd
  ethernets:
    ABC (Use your own interface name):  
      addresses: [192.168.1.1/24] (for instance)
      dhcp4: no
      dhcp6: no
#---------content-----------
```

**❕Each host should have a unique address under the same subnet (e.g. `192.168.1.*`).**

4. Apply the network changes:
```
sudo netplan apply
```

5. Restart network service:

```
sudo systemctl restart systemd-networkd
```

6. Test connection:
```
ping 192.168.1.1 (for instance)
```

#### Restore default connection

1. Reset configuration file by putting back the following:
```
#---------content-----------
network:
  version: 2
  renderer: NetworkManager
#---------content-----------
```
  
2. Apply the network changes:
```
sudo netplan apply
```

3. Restart network service:
```
sudo systemctl restart NetworkManager
```

## 🚨 FLOAT failure detection

The hyperparameters of FLOAT failure detector is included in [our rollout configuration files](./config/).
We detail some key parameters as follows:
```
.
├── failure_detection:
|   ├── max_queue_size: ${the maximum number of requests for failure detection in the asynchronous queue.}
|   └── num_samples: ${number of inferred action chunk samples, default to 1}
|   └── num_expert_candidates: ${number of candidates in expert demonstrations for OT matching}
|   └── ot_percentile: ${the initial percentile of OT values used for threshold computation}
|   └── soft_ot_ratio: ${used for adaptive rewinding}
|   └── update_stats: ${whether to update FLOAT threshold during rollout. Muse be set to True in your first rollout to activate the failure detector!}
|   └── enable_visualization: ${whether to save FLOAT visualizations to output directory}
```

## ▶️ Running multi-robot experiments

We select one of the hosts to be the communication hub of the entire multi-robot system.
We run the following command on the chosen host:

```
python nodes/communication_hub.py
```

Then, we activate the teleoperation nodes via the following command:

```
python nodes/teleop_node.py --teleop_id ${the index of teleoperation node}
```

After that, we run the robot node on every host using the following command:

```
python run_rollout.py --config-name multi_robot_rollout
```

It is worth noting that you need to override the `robot_info` and `camera` attribute in the [configuration file](./config/multi_robot_rollout.yaml) according to your setup.
We give a detailed explanations for the key configuration attributes below:
```
.
├── checkpoint_path: ${pretrained checkpoint path}
├── train_dataset_path: ${training data path including the expert demonstrations}
├── save_buffer_path: ${Output data path containing rollout trajectories}
├── output_dir: ${Output directory for saving scene configurations and FLOAT visualizations}
├── robot_info:
|   ├── num_robot: ${number of robots in parallel}
|   └── socket_ip: ${IP address of the host with communication hub}
|   └── socket_port: ${defined in the hub}
|   └── robot_name: ${flexiv by default}
|   └── robot_id: ${customized for every robot node}
|   └── robot_info_dict: ${must contain the key-value pair for the current robot node. The value should be the local IP address of the robot}
├── camera:
|   ├── serial: 
|   |   ├── ${eye-to-hand camera serial number}
|   |   └── ${eye-in-hand camera serial number}
|   └── fps: ${control frequency}
|   └── img_shape: ${image observation shape}
```