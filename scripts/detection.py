#!/usr/bin/env python3

import rospy
import cv2
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from obj_detection.msg import ObjectDetection

# Import YOLOv5 module
import yolov5

class YOLOV5Detector:
    def __init__(self, weight_file):
        # Load YOLOv5 model
        self.model = yolov5.load(weight_file)

        # Initialize ROS node
        rospy.init_node('yolov5_detector', anonymous=True)

        # Create image subscriber
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('image', Image, self.image_callback)

        # Create object detection publisher
        self.pub = rospy.Publisher('object_detection', ObjectDetection, queue_size=10)

    def image_callback(self, data):
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Run YOLOv5 object detection on image
        results = self.model(cv_image)

        # Create object detection message
        msg = ObjectDetection()
        msg.header.seq = 123
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'camera'

        for label, confidence, bbox in results.xyxy[0]:
            # Add bounding box to message
            box = ObjectDetection.BoundingBox()
            box.xmin = int(bbox[0])
            box.ymin = int(bbox[1])
            box.xmax = int(bbox[2])
            box.ymax = int(bbox[3])
            box.label = label
            box.probability = confidence.item()
            msg.bounding_boxes.append(box)

        # Publish object detection message
        self.pub.publish(msg)

if __name__ == '__main__':
    # Set path to YOLOv5 weight file
    weight_file = '/home/david/yolo_dectction/src/obj_detection/YoloWeights/best.pt'

    # Initialize YOLOv5 detector
    detector = YOLOV5Detector(weight_file)

    # Start ROS node
    rospy.spin()
