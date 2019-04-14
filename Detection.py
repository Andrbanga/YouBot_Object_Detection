
import sys
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import threading
import time


from collections import defaultdict
from io import StringIO
from PIL import Image

import cv2

sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util
print("Import done")

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = '/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
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
        #print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)




if __name__ == "__main__":

    # cap = cv2.VideoCapture("http://192.168.88.22:8080/stream?topic=/camera/rgb/image_raw&width=640&height=480&quality=50")
    cap = cv2.VideoCapture(0)
    width = 640
    height = 480
    midpoint = (width // 2, height // 2)
    cap.set(3, width)
    cap.set(4, height)
    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    ret, image = cap.read()

    lock = threading.RLock()
    def videocap():
        global  image

        while (cap.isOpened()):
            ret, image1 = cap.read()
            with lock:
                image = image1.copy()
            #cv2.imshow("sda", image)
            cv2.waitKey(1)


    def detection():
        while True:
            with lock:
                image2 = image.copy()
            windowNotSet = True
            [h, w] = image2.shape[:2]
            #print (h, w)
            #image2 = cv2.flip(image2, 1)

            (boxes, scores, classes, num_detections) = tDetector.run(image)

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)
            max_boxes_to_draw = boxes.shape[0]

            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     image,
            #     boxes,
            #     classes,
            #     scores,
            #     category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=4)


            for i in range(min(max_boxes_to_draw, boxes.shape[0])):
                if scores is None or scores[i] > .5 and classes[i] == 1:
                    box = tuple(boxes[i].tolist())
                    box = (box[0] * height, box[1] * width, box[2] * height, box[3] * width)
                    box = tuple(map(int, box))
                    ymin, xmin, ymax, xmax = box
                    mid = ((xmax + xmin) // 2, (ymax + ymin) // 2)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (100, 255, 0), 2)

                    cv2.circle(image, mid, 7, (0, 0, 255), 2)
                    print(type(box))
                    # print()
            # print("===================================")
            #################################

            #print(boxes)
            #print(classes)
            #cv2.rectangle(image, (int(boxes[0][0][0]*640), int(boxes[0][0][1]*480)), (int(boxes[0][0][2]*640 ), int(boxes[0][0][3]*480 )),(0, 128, 255), -1)
            if windowNotSet is True:
                cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
                windowNotSet = False

            cv2.imshow("tensorflow based (%d, %d)" % (w, h), image2)

            k = cv2.waitKey(1) & 0xff
            if k == ord('q') or k == 27:
                break


    commander2 = threading.Thread(target=detection)
    commander2.start()

    videocap()

    cap.release()
