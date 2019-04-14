YOUBOT_MODE = False

import numpy as np
import os
import sys
import tensorflow as tf
import threading
import time

import cv2

sys.path.append("..")
from math import sqrt
from utils import label_map_util

print("Import done")

# conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# conn.connect(('192.168.88.22', 7777))

cv2.namedWindow("Video stream", cv2.WINDOW_NORMAL)
if YOUBOT_MODE:
    vc = cv2.VideoCapture(
        "http://192.168.88.22:8080/stream?topic=/camera/rgb/image_raw&width=640&height=480&quality=30")
else:
    vc = cv2.VideoCapture(0)

width = 640
height = 480
midpoint = (width // 2, height // 2)

vc.set(3, width)
vc.set(4, height)
vc.set(cv2.CAP_PROP_EXPOSURE, -3)  # old: 5

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

tracker = cv2.TrackerBoosting_create()
ret, img = vc.read()
connectionIsOpen = False

lock = threading.RLock()

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = '/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

if True:  # move to different file
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


    class TensoflowFaceDector(object):
        def __init__(self, PATH_TO_CKPT):
            """Tensorflow detector
            """

            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            with self.detection_graph.as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(graph=self.detection_graph, config=config)
                self.windowNotSet = True

        def run(self, image):
            """image: bgr image
            return (boxes, scores, classes, num_detections)
            """

            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            start_time = time.time()
            (boxes, scores, classes, num_detections) = self.sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            elapsed_time = time.time() - start_time
            # print('inference time cost: {}'.format(elapsed_time))

            return (boxes, scores, classes, num_detections)


class Kuka:
    def __init__(self, conn):
        self.conn = conn

    def send_data(self, data):  # отправка данных на робота
        self.conn.send(data)
        print('Data sent: ' + data.decode())

    def receive_data(self):  # Получение одометрии
        data = ''
        while connectionIsOpen:
            rcvd_data = self.conn.recv(1)
            if rcvd_data.decode() == '\n':
                print(data)
                data = ''
            else:
                data += rcvd_data.decode()

    def SendCommand(self):
        global vecX
        global vecY
        global initX
        global initY
        global initM
        # time.sleep(1) # FIX IT
        connectionIsOpen = True
        # receive_thread = threading.Thread(target=receive_data)
        # receive_thread.start(

        while True:

            print("%d, %d" % (vecX, vecY))
            if (vecX > 5 or vecX < -5):
                requiredTimeLocal = sqrt(vecX ** 2 + vecY ** 2) * movementSpeed
                initX = initX + vecX
                initX = clamp(initX, 80, 256)
                initY = initY + vecY
                initY = clamp(initY, 10, 177)
                self.send_data(b'LUA_ManipDeg(0, %d, 10, -83, %d, %d)^^^' % (initX, initY, initM))
                print(b'LUA_ManipDeg(0, %d, 61, -139, %d, %d)^^^' % (initX, initY, initM))

                time.sleep(requiredTimeLocal)

            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break


tDetector = TensoflowFaceDector(PATH_TO_CKPT)


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
        if key == 27:  # exit on ESC
            # connectionIsOpen = False
            # send_data(b'#end#^^^')
            # conn.shutdown(socket.SHUT_RDWR)
            # conn.close()
            break
        # if key == 119:
        #     send_data(b'LUA_Base(0.1, 0, 0)^^^')
        # if key == 113:
        #     send_data(b'LUA_Base(0, 0, 0)^^^')
        # if key == 91:
        #     initM = 190
        #     send_data(b'LUA_ManipDeg(0, %d, 10, -83, %d, %d)^^^' % (initX, initY, initM))
        # if key == 93:
        #     initM = 170
        #     send_data(b'LUA_ManipDeg(0, %d, 10, -83, %d, %d)^^^' % (initX, initY, initM))
        #     send_data(b'LUA_Gripper(0, 0.3)^^^')


def detection():
    global lastPoint
    global delta
    global initX
    global vecX
    global vecY
    global h
    box = [midpoint[0], midpoint[1], midpoint[0] + 1, midpoint[1] + 1]
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
            if scores is None or scores[i] > .7 and classes[i] == 1:
                box = tuple(boxes[i].tolist())
                box = (box[0] * height, box[1] * width, box[2] * height, box[3] * width)
                box = tuple(map(int, box))

                if box is not None:
                    # points = box
                    # ok = tracker.init(frame, (points[1], points[0], points[3], points[2]))
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
                    fixX = (lastPoint[0] - midpoint[0]) * completionK
                    fixY = (lastPoint[1] - midpoint[1]) * completionK
                    # nearestPoint = (nearestPoint[0] - fixX // 2, nearestPoint[1] - fixY // 2)
                # else:
                #     #curX = clamp(lastPoint[0] + delta[0] / 2, 0, width / 2)
                #     #curY = clamp(lastPoint[1] + delta[1] / 2, 1, height / 2)
                #     #nearestPoint = (lastPoint[0] , lastPoint[1] )
                #     nearestPoint = midpoint
                #     delta = (nearestPoint[1] - lastPoint[0], nearestPoint[0] - lastPoint[1])
                #     lastPoint = nearestPoint

            vecX = int((nearestPoint[0] - midpoint[0]) * ratio)
            vecY = int((nearestPoint[1] - midpoint[1]) * ratio)
            cv2.imshow("Video stream", frame)

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break


commander2 = threading.Thread(target=detection)
commander2.start()

# commander1 = threading.Thread(target=SendCommand)
# commander1.start()
#
# commander3 = threading.Thread(target=BaseControl)
# commander3.start()

# send_data(b'LUA_ManipDeg(0, 168, 10, -83, 177, 170)^^^') # манипулятор в изначально положение

videocap()

cv2.destroyWindow("Video stream")
