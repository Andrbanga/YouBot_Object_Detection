import time
from math import sqrt
import cv2



class Kuka:

    def clamp(self, value, minval, maxval):
        return self.sorted((value, minval, maxval))[1]

    def signum(self, val):
        if val < 0:
            return -1
        if val > 0:
            return 1
        if val == 0:
            return 0


    def __init__(self, conn, connectionIsOpen):
        self.conn = conn
        self.connectionIsOpen = connectionIsOpen

    def send_data(self, data):  # отправка данных на робота
        self.conn.send(data)
        print('Data sent: ' + data.decode())

    def receive_data(self):  # Получение одометрии
        data = ''
        while self.connectionIsOpen:
            rcvd_data = self.conn.recv(1)
            if rcvd_data.decode() == '\n':
                print(data)
                data = ''
            else:
                data += rcvd_data.decode()

    def SendCommand(self, vecX, vecY, initX, initY, initM, movementSpeed):
        self.connectionIsOpen = True
        # receive_thread = threading.Thread(target=receive_data)
        # receive_thread.start(

        while True:

            print("%d, %d" % (vecX, vecY))
            if (sqrt(vecX) + sqrt(vecY) < 16):
                requiredTimeLocal = sqrt(vecX ** 2 + vecY ** 2) * movementSpeed
                initX = initX + self.signum(vecX)
                initX = self.clamp(initX, 80, 256)
                initY = initY + self.signum(vecY)
                initY = self.clamp(initY, 10, 177)
                self.send_data(b'LUA_ManipDeg(0, %d, 10, -83, %d, %d)^^^' % (initX, initY, initM))
                print(b'LUA_ManipDeg(0, %d, 61, -139, %d, %d)^^^' % (initX, initY, initM))

                time.sleep(requiredTimeLocal)

            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
