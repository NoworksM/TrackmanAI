# standard library imports
import json
import math
import os
import socket
import struct
import time
from pathlib import Path
from threading import Lock, Thread

# third-party imports
import numpy as np


class TM2020OpenPlanetClient:
    def __init__(self, host='127.0.0.1', port=9000, struct_str='<' + 'f' * 11):
        self._struct_str = struct_str
        self.nb_floats = self._struct_str.count('f')
        self.nb_uint64 = self._struct_str.count('Q')
        self._nb_bytes = self.nb_floats * 4 + self.nb_uint64 * 8

        self._host = host
        self._port = port

        # Threading attributes:
        self.__lock = Lock()
        self.__data = None
        self.__t_client = Thread(target=self.__client_thread, args=(), kwargs={}, daemon=True)
        self.__t_client.start()

    def __client_thread(self):
        """
        Thread of the client.
        This listens for incoming data until the object is destroyed
        TODO: handle disconnection
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self._host, self._port))
            data_raw = b''
            while True:  # main loop
                while len(data_raw) < self._nb_bytes:
                    data_raw += s.recv(1024)
                div = len(data_raw) // self._nb_bytes
                data_used = data_raw[(div - 1) * self._nb_bytes:div * self._nb_bytes]
                data_raw = data_raw[div * self._nb_bytes:]
                self.__lock.acquire()
                self.__data = data_used
                self.__lock.release()

    def get_data(self, sleep_if_empty=0.01, timeout=10.0):
        """
        Retrieves the most recently received data
        Use this function to retrieve the most recently received data
        This blocks if nothing has been received so far
        """
        c = True
        t_start = None
        while c:
            self.__lock.acquire()
            if self.__data is not None:
                data = struct.unpack(self._struct_str, self.__data)
                c = False
                self.__data = None
            self.__lock.release()
            if c:
                if t_start is None:
                    t_start = time.time()
                t_now = time.time()
                assert t_now - t_start < timeout, f"OpenPlanet stopped sending data since more than {timeout}s."
                time.sleep(sleep_if_empty)
        return Trackmania2020Data(data)


class Trackmania2020Data:
    def __init__(self, data):
        self.x: float = data[2]
        self.y: float = data[3]
        self.z: float = data[4]
        self.terminated: bool = bool(data[8])
        self.speed = data[0]
        self.distance = data[1]
        self.steering_input: float = data[5]
        self.accelerate: float = data[6]
        self.brake: float = data[7]
        self.gear = data[9]
        self.rpm: int = data[10]

    def __str__(self):
        return (f'x: {self.x}, y: {self.y}, z: {self.z}, terminated: {self.terminated}, '
                f'steering_input: {self.steering_input}, accelerate: {self.accelerate}, '
                f'brake: {self.brake}, rpm: {self.rpm}, speed: {self.speed}')


def save_ghost(host='127.0.0.1', port=10000):
    """
    Saves the current ghost

    Args:
        host (str): IP address of the ghost-saving server
        port (int): Port of the ghost-saving server
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))


def armin(tab):
    nz = np.nonzero(tab)[0]
    if len(nz) != 0:
        return nz[0].item()
    else:
        return len(tab) - 1
