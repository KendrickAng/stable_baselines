import socket
import pickle

class NavInterface:
    """
    This class sends actions to the car via a socket
    """
    def __init__(self):
        self._socket = socket.socket()
        socket_addr = ('127.0.0.1', 8888)
        self._socket.connect(socket_addr)

    def send_data(self, output_data):
        data = [output_data]
        data_string = pickle.dumps(data, protocol=1)
        self._socket.send(data_string)

    def quit(self):
        self._socket.close()