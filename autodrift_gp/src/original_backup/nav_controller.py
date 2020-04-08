#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

import socket
import pickle

class NavController:
    def __init__(self):
        self._cmd_vel_nav_pub = rospy.Publisher('/cmd_vel_nav', Twist, queue_size=1)

        self._socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('127.0.0.1', 8888)
        self._socket_server.bind(server_address)
        self._socket_server.listen(5)
        self._connection, client_address = self._socket_server.accept()

        self.run()

    def run(self):
        while True:
            data = self._connection.recv(1024)
            if data:
                try:
                    msg = pickle.loads(data)
                    # NOTE: the rest of the code in this section has been deliberately omitted

                except Exception as e:
                    # print('NavController: socket exception', e)
                    pass
            else:
                break

        self._connection.close()


if __name__ == "__main__":
    rospy.init_node('nav_controller')
    NavController()
    rospy.spin()


