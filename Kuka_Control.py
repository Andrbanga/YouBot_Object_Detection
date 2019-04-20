import time
from math import sqrt
import cv2
import threading



class Kuka(threading.Thread):

    def clamp(self, value, minval, maxval):
        return self.sorted((value, minval, maxval))[1]




    def __init__(self, conn, connectionIsOpen):
        self.conn = conn
        self.connectionIsOpen = connectionIsOpen


