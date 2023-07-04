import queue
import socket
from threading import Thread


class IpcServer(object):
    """Ipc server for multi processing communication, and the backend is socket
    It has a queue for storing messages which are from Client."""

    _instance = None

    def __init__(self):

        self.queue = queue.Queue()
        # create socket and bind a port
        self.ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ss.bind(('', 0))
        self.ss.listen(128)
        self.ss.settimeout(5)
        self.running = True
        # start listening
        self.start_listen_thread = Thread(target=self.start_listen)
        self.start_listen_thread.setDaemon(True)
        self.start_listen_thread.start()
        # ip and port
        self.port = self.ss.getsockname()[1]
        self.ip = socket.gethostbyname(socket.gethostname())

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def start_listen(self):
        while self.running:
            try:
                conn, _ = self.ss.accept()
            except Exception:
                continue
            data = conn.recv(4096)
            conn.close()
            data = data.decode()
            self.queue.put(data)

    def close(self):
        self.running = False
        self.ss.close()
        self.start_listen_thread.join()

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        self.close()

    def __del__(self):
        self.close()


class IpcClient(object):
    """Ipc client for sending message to server, and will close the connection
    after sending messages."""

    @classmethod
    def send(self, ip, port, data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ss:
            ss.connect((ip, int(port)))
            data = str.encode(data)
            ss.send(data)
            ss.close()
