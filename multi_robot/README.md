## Basic Usage

### Network Setup (via Ethernet)

#### A.SETUP DIRECT CONNECTION

1. use a Ethernet cable to connect 2 computers.

2. check the wired network interface

```
ip a
```
The name of the second interface should overwrite the ABC values below.

3. configure static IP addresses

```
sudo gedit /etc/netplan/01-network-manager-all.yaml
```

Put the following content into the configuration file


```

#---------content-----------
network:
  version: 2
  renderer: networkd
  ethernets:
    ABC (Use your own interface name):  
      addresses: [192.168.1.1/24] for A and [192.168.1.2/24] for B
      dhcp4: no
      dhcp6: no
#---------content-----------
```

4. apply the changes


```
sudo netplan apply
```

5. verify service status

```
systemctl status systemd-networkd # Optional
sudo systemctl restart systemd-networkd  # If service shows errors
```

6. test the connection 
```
ping 192.168.1.2 (on A) 
ping 192.168.1.1 (on B)
```

#### B. RESTORE DEFAULT CONNECTION

1. reset configuration

Put back to the configuration file:
```
#---------content-----------
network:
  version: 2
  renderer: NetworkManager
#---------content-----------
```
  
2. apply the changes
```
sudo netplan apply
```


3. restart NetworkManager
```
systemctl status NetworkManager
sudo systemctl restart NetworkManager  # If service shows errors
```

### Multi-robot commands


```
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd multi_robot

python nodes/communication_hub.py
python nodes/teleop_node.py --teleop_id 0
python run_real_rollout.py --config-name multi_robot_config
```