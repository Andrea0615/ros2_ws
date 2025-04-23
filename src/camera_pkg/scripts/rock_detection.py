#!/usr/bin/env python
import os
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D
from ultralytics import YOLO
from primesense import openni2

class StoneDetectorNode:
    def __init__(self):
        rospy.init_node("stone_detector")

        # --- Parameters ---
        base_dir = rospy.get_param("~base_dir", os.path.dirname(__file__))
        openni2_path = os.path.join(base_dir, "OpenNI2_Redist")
        yolo_weights = rospy.get_param(
            "~yolo_weights",
            os.path.join(base_dir, "cameraTrainingV2/cameraTraining/runs/detect/train2/weights/last.pt")
        )
        cam_index = rospy.get_param("~camera_index", 0)

        # --- Verify paths ---
        if not os.path.isdir(openni2_path):
            rospy.logerr("OpenNI2 path not found: %s", openni2_path)
            rospy.signal_shutdown("Missing OpenNI2")
        if not os.path.isfile(yolo_weights):
            rospy.logerr("YOLO weights not found: %s", yolo_weights)
            rospy.signal_shutdown("Missing YOLO weights")

        # --- Initialize OpenNI2 ---
        openni2.initialize(openni2_path)
        self.dev = openni2.Device.open_any()
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.start()

        # --- Load YOLO model ---
        self.model = YOLO(yolo_weights)

        # --- OpenCV camera ---
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            rospy.logerr("Cannot open camera index %d", cam_index)
            rospy.signal_shutdown("Camera failed")

        # --- Publishers ---
        self.bridge = CvBridge()
        self.img_pub = rospy.Publisher("/detection_image", Image, queue_size=1)
        self.det_pub = rospy.Publisher("/detections", Detection2DArray, queue_size=1)

        rospy.loginfo("StoneDetectorNode initialized, starting loop...")
        self.loop()

    def enhance_contrast(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    def loop(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("Empty frame")
                continue

            frame = self.enhance_contrast(frame)
            resized = cv2.resize(frame, (640, 480))
            results = self.model(resized)

            # read depth
            frm = self.depth_stream.read_frame()
            depth_arr = np.frombuffer(frm.get_buffer_as_uint16(),
                                      dtype=np.uint16).reshape((frm.height, frm.width))

            det_array = Detection2DArray()
            det_array.header.stamp = rospy.Time.now()
            det_array.header.frame_id = "camera_rgb_optical_frame"

            # process each detection
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                cx, cy = (x1+x2)//2, (y1+y2)//2

                # depth lookup (check bounds)
                if 0 <= cy < depth_arr.shape[0] and 0 <= cx < depth_arr.shape[1]:
                    z = depth_arr[cy, cx] / 1000.0
                else:
                    z = float('nan')

                # draw
                cv2.rectangle(resized, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.circle(resized, (cx, cy), 4, (0,0,255), -1)
                label = f"x={cx},y={cy},z={z:.2f}m"
                cv2.putText(resized, label, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

                # build Detection2D
                det = Detection2D()
                det.header = det_array.header
                det.results = [ ObjectHypothesisWithPose(id=str(box.cls.cpu().item()),
                                                        score=box.conf.cpu().item()) ]
                det.bbox.center.x = float(cx)
                det.bbox.center.y = float(cy)
                det.bbox.size_x = float(x2 - x1)
                det.bbox.size_y = float(y2 - y1)
                # encode depth in pose.theta
                det.bbox.center.theta = z
                det_array.detections.append(det)

            # publish annotated image
            img_msg = self.bridge.cv2_to_imgmsg(resized, encoding="bgr8")
            img_msg.header = det_array.header
            self.img_pub.publish(img_msg)

            # publish detections
            self.det_pub.publish(det_array)

            rate.sleep()

        # clean up
        self.depth_stream.stop()
        openni2.unload()
        self.cap.release()


if __name__ == "__main__":
    try:
        StoneDetectorNode()
    except rospy.ROSInterruptException:
        pass