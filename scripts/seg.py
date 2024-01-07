#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sensor_msgs.msg import Image
# import rosbag

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.InteractiveSession(config=config)
# K.set_session(sess)



class SemanticSegmentationNode:
    def __init__(self):
        rospy.init_node('semantic_segmentation_node')
        
        self.bridge = CvBridge()

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices)>0:
            tf.config.experimental.set_memory_growth(physical_devices[0],True)

        self.rate = rospy.Rate(30)

        self.model = load_model('/home/jetson/catkin_ws/src/ros_segmentation/model/unet_class1_v03.hdf5')
        
        self.publisher = rospy.Publisher('/segmentation_result_topic', Image, queue_size=1)
        self.subscriber = rospy.Subscriber('/rgbd_image_topic', Image, self.callback)
        
        # self.process_bag(bag_file)
    
    def segment_image(self, image):
        # Perform segmentation using your Keras model
        segmentation_result = self.model.predict(image)

        # Post-process the segmentation result as needed
        postprocessed_result = self.postprocess_result(segmentation_result)

        return postprocessed_result

    def postprocess_result(self, segmentation_result):
        # Implement any postprocessing steps needed for your segmentation result
        # This may include thresholding, color mapping, etc.
        # Example: thresholding the result
        thresholded_result = (segmentation_result > 0.4).astype(np.uint8) * 255
        resized_img = cv2.resize(thresholded_result, (512, 424))
        return resized_img

    def callback(self, point_cloud_msg):
        rospy.loginfo("RGBD topic is subscribed. Inference has started.")
        processed_img = self.bridge.imgmsg_to_cv2(point_cloud_msg, desired_encoding = 'passthrough')
        segmented_image = self.segment_image(processed_img)
        segmentation_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding='bgra8')
        self.publisher.publish(segmentation_msg)
        self.rate.sleep()

if __name__ == '__main__':
    # bag_file_path = '/home/kojima/ros_bags/rlm_rosbag_2023_11_07-13_21_16_2023_11_07-16_32_38.bag'
    # node = SemanticSegmentationNode(bag_file_path)
    node = SemanticSegmentationNode()
    rospy.spin()
    