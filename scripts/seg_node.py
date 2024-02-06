#!/usr/bin/env python3

import sys 
sys.path.append('/home/kojima/catkin_ws/src/ros_segmentation')
import skimage.transform as transform

import os

import rospy
import pcl
import cv2 as cv
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pcl2

from scripts.model import bce_dice_loss, dice_coef
import scripts.model

class SemanticSegmentationNode:
    def __init__(self):
        #self.model = scripts.model.load_model("/home/kojima/catkin_ws/src/ros_segmentation/model/unet_class1_v07.h5", custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef},compile=False)
        self.model = scripts.model.load_model("/home/kojima/catkin_ws/src/ros_segmentation/model/unet_test", custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef},compile=False)
        
        self.bridge = CvBridge()
        
        rospy.loginfo("Model initialized")

        self.rgbd_img = None
        self.publisher = rospy.Publisher("/segmentation_result_topic", Image, queue_size=1)
        self.subscriber = rospy.Subscriber("/rgbd_image_topic", Image, self.callback)
        self.prediction = False
        rospy.loginfo("ROS node has been initialized.")
        rospy.wait_for_service("/rgbd_image_topic")

    def postprocess_image(self, result):
        seg_result = result.reshape(424,424,1)
        bin_mask = (seg_result > 0.5).astype(np.uint8)
        bin_mask = bin_mask*255
        bin_mask = cv.resize(bin_mask,(512,424))
        return bin_mask
        
        
    def predict(self, image):
        results = self.model.predict(image, steps = 1, verbose = 1)
        self.prediction = False
        return results
    
    def callback(self, msg):
        if not self.prediction:
            try:
                rgbd_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding = "passthrough")
            except CvBridgeError as e:
                print(e)
            
            print("input image shape",rgbd_img.shape)
            print("input image type",rgbd_img.dtype)
            rgbd_img = rgbd_img/255
            rgbd_img = transform.resize(rgbd_img, (424,424))
            rgbd_img = np.expand_dims(rgbd_img, axis = 0)
            print("preprocessed image shape",rgbd_img.shape)
            self.rgbd_img = rgbd_img
            self.prediction = True
            
            if self.prediction:
                rospy.loginfo("Test image has been loaded.")
                rospy.loginfo("Prediction will start...")
                
                test_img= self.rgbd_img

                rospy.loginfo("Predicting the object...")
                result = self.predict(test_img)
                predicted_mask = self.postprocess_image(result)

                print("output shape", np.shape(predicted_mask))
               
                mask_msg = self.bridge.cv2_to_imgmsg(predicted_mask)
                
                self.publisher.publish(mask_msg)
                rospy.loginfo("Publishing the mask")
                self.prediction = False
                


if __name__ == '__main__':
    rospy.init_node('semantic_segmentation_node')
    node = SemanticSegmentationNode()
    rospy.spin()
    
