#!/usr/bin/env python3

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import load_model

import sys 
sys.path.append('/home/kojima/catkin_ws/src/ros_segmentation')
import skimage.transform as trans

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
        self.model = scripts.model.load_model("/home/kojima/catkin_ws/src/ros_segmentation/model/unet_test", custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef})
        
        self.bridge = CvBridge()
        
        rospy.loginfo("Model initialized")
        # physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # if len(physical_devices)>0:
        #     tf.config.experimental.set_memory_growth(physical_devices[0],True)
        #self.rate = rospy.Rate(30)
        #self.rgbd_img = None
        self.img_and_stamp = None
        # self.context =  self.engine.create_execution_context()
        # self.input_cpu, self.input_gpu, self.output_gpu = self.allocate_buffer(self.engine)
        #self.model = load_model('/home/jetson/catkin_ws/src/ros_segmentation/model/unet_class1_v03.hdf5')
        #self.bindings, self.input_buffer, self.input_gpu, self.output_buffer, self.output_gpu = self.buffer(self.rgbd_img, self.engine)
        self.publisher = rospy.Publisher("/segmentation_result_topic", Image, queue_size=1)
        #self.subscriber = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.callback)
        self.subscriber = rospy.Subscriber("/rgbd_image_topic", Image, self.callback)
        self.prediction = False
        rospy.loginfo("ROS node has been initialized.")
        rospy.wait_for_service("/rgbd_image_topic")

    def postprocess_image(self, result):
        seg_result = result.reshape(424,424,1)
        print("postprocess_image max", np.max(seg_result))
        print("postprocess_image min", np.min(seg_result))
        bin_mask = (seg_result > 0.5).astype(np.uint8)
        bin_mask = bin_mask*255
        bin_mask = cv.resize(bin_mask,(512,424))
        return bin_mask
        
        
    def predict(self, image):
        results = self.model.predict(image, steps = 1, verbose = 1)
        print(f"predict min {np.min(results)}")
        print(f"predict max {np.max(results)}")
        self.prediction = False
        return results
    
    def callback(self, msg):
        if not self.prediction:
            try:
                rgbd_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding = "passthrough")
            except CvBridgeError as e:
                print(e)
            #print(np.shape(msg))
            
            print("input image shape",rgbd_img.shape)
            print("input image type",rgbd_img.dtype)
            rgbd_img = rgbd_img/255
            
            rgbd_img = trans.resize(rgbd_img, (424,424))
            print("input max", np.max(rgbd_img))
            print("input min", np.min(rgbd_img))
            cv.imshow("test", rgbd_img)
            #rgbd_img = np.array(rgbd_img, dtype = np.float32)
            rgbd_img = np.expand_dims(rgbd_img, axis = 0)
            #rgbd_img = np.reshape(rgbd_img,(1,) +rgbd_img.shape)
            print("preprocessed image shape",rgbd_img.shape)
            self.img_and_stamp = rgbd_img
            self.prediction = True
            
            if self.prediction:
                rospy.loginfo("Test image has been loaded.")
                rospy.loginfo("Prediction will start...")
                
                test_img= self.img_and_stamp

                rospy.loginfo("Predicting the object...")
                result = self.predict(test_img)
                predicted_mask = self.postprocess_image(result)
                #print(predicted_mask)
                print(f"predicted_mask min {np.min(predicted_mask)}")
                print(f"predicted_mask max {np.max(predicted_mask)}")
                print("output shape", np.shape(predicted_mask))
               
                mask_msg = self.bridge.cv2_to_imgmsg(predicted_mask)
                
                self.publisher.publish(mask_msg)
                rospy.loginfo("Publishing the mask")
                #plt.imshow(predicted_mask,cmap='gray')
                cv.imshow("predicted mask", predicted_mask)
                cv.waitKey(3)
               # cv.destroyAllWindows()
                #plt.show()
                self.prediction = False
                


if __name__ == '__main__':
    rospy.init_node('semantic_segmentation_node')
    node = SemanticSegmentationNode()
    rospy.spin()
    
