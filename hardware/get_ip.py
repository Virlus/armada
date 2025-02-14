def get_local_ip(ip='8.8.8.8'):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((ip, 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

def get_ip():
    robot_ip = "192.168.2.100"
    local_ip = get_local_ip(robot_ip)
    return robot_ip,local_ip

if __name__ == "__main__":
    robot_ip = "192.168.2.100"
    local_ip = get_local_ip(robot_ip)
    print(robot_ip,local_ip)