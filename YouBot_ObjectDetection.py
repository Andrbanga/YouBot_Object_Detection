# -*- coding: utf-8 -*-
import socket
import numpy as np
import os
import threading
import time
import cv2
from math import sqrt
from Brain import TensoflowFaceDector
from Kuka_Control import Kuka

YOUBOT_MODE = False
print("Import done")



Youbot_hostname = "192.168.88.22"
Youbot_host = os.system("ping -n 1 " + Youbot_hostname)

if Youbot_host == 0:
    YOUBOT_MODE = True
    print("YouBot Mode")
else:
    YOUBOT_MODE = False
    print("Debug Mode")

cv2.namedWindow("Video stream", cv2.WINDOW_NORMAL)

if YOUBOT_MODE:
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect((Youbot_hostname, 7777))
    connectionIsOpen = True
    kuka = Kuka(conn, connectionIsOpen)
    vc = cv2.VideoCapture(
        "http://192.168.88.22:8080/stream?topic=/camera/rgb/image_raw&width=640&height=480&quality=30")
else:
    vc = cv2.VideoCapture(0)

width = 640
height = 480
midpoint = (width // 2, height // 2)

vc.set(3, width)
vc.set(4, height)
vc.set(cv2.CAP_PROP_EXPOSURE, -6)  # old: 5

FOVx = 58  # degree
FOVy = 45
ratio = FOVx / width  # degree per pixel

movementSpeed = 0.00411 / ratio

lastPoint = midpoint
delta = (0, 0)
minDelta = 2

pos = 0
vecX = 0
vecY = 0
initX = 168
initY = 176
initM = 170


ret, img = vc.read()
lock = threading.RLock()



def drawRect(whpoint):
    ymin, xmin, ymax, xmax = whpoint
    mid = ((xmax + xmin) // 2, ymin + 30)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.circle(img, mid, 7, (0, 0, 255), 2)


def get_distance(point):
    return (midpoint[0] - point[1]) ** 2 + (midpoint[1] - point[0]) ** 2


def get_nearest_point(points):
    nearestDist = 100000
    nearestPoint = points[1]
    dist = get_distance(points)
    if dist < nearestDist:
        nearestDist = dist
        nearestPoint = points
        return nearestPoint
    else:
        nearestPoint = [midpoint[1], midpoint[0], midpoint[1], midpoint[0]]
        return nearestPoint


def clamp(value, minval, maxval):
    return sorted((value, minval, maxval))[1]


def round_tuple(items):  # sorry, I know I must use normal vectors
    return tuple(map(int, map(round, items)))


def videocap():
    global img
    global initM
    global initX
    global initY
    while (vc.isOpened()):
        ret, img_ = vc.read()
        with lock:
            img = img_.copy()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC]
            if YOUBOT_MODE:
                connectionIsOpen = False
                kuka.send_data(b'#end#^^^')
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
            break
        if YOUBOT_MODE:
            if key == 119:
                kuka.send_data(b'LUA_Base(0.1, 0, 0)^^^')
            if key == 113:
                kuka.send_data(b'LUA_Base(0, 0, 0)^^^')
            if key == 91:
                initM = 190
                kuka.send_data(b'LUA_ManipDeg(0, %d, 10, -83, %d, %d)^^^' % (initX, initY, initM))
            if key == 93:
                initM = 170
                kuka.send_data(b'LUA_ManipDeg(0, %d, 10, -83, %d, %d)^^^' % (initX, initY, initM))
                kuka.send_data(b'LUA_Gripper(0, 0.3)^^^')


def detection():
    # todo: move to brain and remove globals
    # global lastPoint
    # global delta
    # global initX
    # box = [midpoint[0], midpoint[1], midpoint[0] + 1, midpoint[1] + 1]
    tDetector = TensoflowFaceDector()
    while (ret == True):
        with lock:
            frame = img.copy()

        (boxes, scores, classes, num_detections) = tDetector.run(frame)
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        max_boxes_to_draw = boxes.shape[0]
        nearestPoint = midpoint
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > .5 and classes[i] == 1:
                box = tuple(boxes[i].tolist())
                box = (box[0] * height, box[1] * width, box[2] * height, box[3] * width)
                box = tuple(map(int, box))

                if box is not None:
                    # points = box
                    nearestPoint = get_nearest_point(box)
                    # nearestPoint = points
                    drawRect(nearestPoint)
                    nearestPoint = ((nearestPoint[1] + nearestPoint[3]) / 2, nearestPoint[0] + 30)
                    cv2.line(frame, round_tuple(midpoint), round_tuple(nearestPoint), (232, 244, 66), 2)
                    # approx
                    elapsedTimeSinceLastCommand = time.time()
                    lastCommandTime = time.time()

                    distance = sqrt(get_distance(nearestPoint))
                    requiredTime = distance / movementSpeed
                    if requiredTime == 0:
                        completionK = 1
                    else:
                        completionK = clamp(elapsedTimeSinceLastCommand / requiredTime, 0, 1)
                    # fixX = (lastPoint[0] - midpoint[0]) * completionK
                    # fixY = (lastPoint[1] - midpoint[1]) * completionK
                    vecX = int((nearestPoint[0] - midpoint[0]) * ratio)
                    vecY = int((nearestPoint[1] - midpoint[1]) * ratio)
                    print(vecY, vecY, scores[i])

                    # nearestPoint = (nearestPoint[0] - fixX // 2, nearestPoint[1] - fixY // 2)
                # else:
                #     #curX = clamp(lastPoint[0] + delta[0] / 2, 0, width / 2)
                #     #curY = clamp(lastPoint[1] + delta[1] / 2, 1, height / 2)
                #     #nearestPoint = (lastPoint[0] , lastPoint[1] )
                #     nearestPoint = midpoint
                #     delta = (nearestPoint[1] - lastPoint[0], nearestPoint[0] - lastPoint[1])
                #     lastPoint = nearestPoint


        cv2.imshow("Video stream", frame)

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break


commander2 = threading.Thread(target=detection)
commander2.start()

if YOUBOT_MODE:
    commander1 = threading.Thread(target=Kuka.SendCommand, args=(vecX, vecY, initX, initY, initM, movementSpeed))
    commander1.start()

    kuka.send_data(b'LUA_ManipDeg(0, 168, 10, -83, 177, 170)^^^') # манипулятор в изначально положение

videocap()

cv2.destroyWindow("Video stream")
