#!/usr/bin/env python3

import os
import sys 
sys.path.append('/home/jetson/catkin_ws/src/ros_segmentation')
# import pycuda.driver as cuda
# import pycuda.autoinit
# import tensorrt as trt
import rospy
from scripts.inference import get_TRT, load_trtengine
import cv2 as cv
import numpy as np
# import sensor_msgs
# import sensor_msgs.point_cloud2 as pcl2
# from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge, CvBridgeError
# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import load_model
from sensor_msgs.msg import Image
# import rosbag
import skimage.transform as trans
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.InteractiveSession(config=config)
# K.set_session(sess)


engine_file = "/home/jetson/catkin_ws/src/ros_segmentation/model/unet/model_seg.engine"


class SemanticSegmentationNode:
    def __init__(self):
        self.engine = load_trtengine(engine_file)
        self.TRT = get_TRT(self.engine)

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
        self.publisher = rospy.Publisher("/segmentation_result_topic", Image, queue_size=10)
        #self.subscriber = rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.callback)
        self.subscriber = rospy.Subscriber("/rgbd_image_topic", Image, self.callback)
        self.prediction = False
        rospy.loginfo("ROS node has been initialized.")
        rospy.wait_for_service("/rgbd_image_topic")
        
        
        
    # def load_trtengine(self, engine):
    #     print("Loading TRT engine from file :",engine)
    #     with open(engine, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    #         engine = runtime.deserialize_cuda_engine(f.read())
    #         return engine


        
    # def preprocess_image(self, cloud):
    #     cloud_data = pcl2.read_points(cloud, field_names=("x", "y", "z","rgb"),skip_nans = True)
    #     cloud_array = np.array(list(cloud_data))
    #     print(np.shape(cloud_array))
    #     # rgb_img = cloud_array[:,3].astype(np.uint32)
    #     # #rgb_img = self.bridge.imgmsg_to_cv2(cloud_array, desired_encoding = 'passthrough')
    #     # self.rgbd_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2BGRA)
    #     depth_img = np.sqrt(np.sum(cloud_array[:, :3]**2, axis = 1))
    #     depth_img = cv.normalize(depth_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    #     cv.imshow("depth",depth_img)
    #     self.rgbd_img[:, :, 3] = depth_img
    #     preprocessed_img = self.rgbd_img/255
    #     preprocessed_img = np.resize(preprocessed_img, (424, 424))
    #     cv.imshow("rgbd",preprocessed_img)
    #     return preprocessed_img

    def postprocess_image(self, result):
        result = np.reshape(result, (424, 424))
        
        mask = (result > 0.5).astype(np.uint8)*255
        mask = np.reshape(mask, (512, 424))
        # resized_img = cv.resize(thresholded_result, (512, 424))
        return mask

    def callback(self, msg):
        if not self.prediction:
            try:
                rgbd_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding = "passthrough")
            except CvBridgeError as e:
                print(e)
            #print(np.shape(msg))
            rgbd_img = rgbd_img/255
            rgbd_img = trans.resize(rgbd_img, (424,424))
            rgbd_img = np.array(rgbd_img, dtype = np.float32)
            rgbd_img = np.expand_dims(rgbd_img, axis = 0)
            #rgbd_img = np.reshape(rgbd_img,(1,) +rgbd_img.shape).astype(np.float32)
            # rgbd_img = cv.resize(rgbd_img, (424, 424)).astype(np.float32)
            self.img_and_stamp = rgbd_img
            self.prediction = True
            rospy.loginfo("Test image has been loaded.")
            if self.prediction:
                test_img= self.img_and_stamp
                print(np.shape(test_img))
                rospy.loginfo("Predicting the object...")
                result = self.TRT.predict(test_img)
                print(np.shape(result))
                predicted_mask = self.postprocess_image(result)
                #print(predicted_mask)
                print(np.shape(predicted_mask))
                
                mask_msg = self.bridge.cv2_to_imgmsg(predicted_mask, "rgb8")
                #mask_msg.header.stamp = timestamp
                print(np.shape(mask_msg))
                self.publisher.publish(mask_msg)
                rospy.loginfo("Publishing the mask")
                cv.imshow("predicted mask", predicted_mask)
                cv.waitKey()
                self.prediction = False
        
        
        #self.rgbd_img = np.reshape(self.rgbd_img.shape).astype(np.float32)
        #print(np.shape(self.rgbd_img))
        # cv.imshow("rgbd",self.rgbd_img)
        # cv.waitKey()

        # loaded_engine = self.loadTRT(self.engine)
        # rospy.loginfo("TRT engine has started")
        # rgbd = self.preprocess_image(point_cloud_msg)
        # rospy.loginfo("RGBD image has been obtained. Inference has started")
if __name__ == '__main__':
    rospy.init_node('semantic_segmentation_node')
    node = SemanticSegmentationNode()
    rospy.spin()
    