#!/usr/bin/env python3

import onnxruntime as ort
import skimage.transform as trans

import os

import rospy

import cv2 as cv
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image



class SemanticSegmentationNode:
    def __init__(self):
        self.model = '/home/jetson/catkin_ws/src/ros_segmentation/model/unet/model.onnx'
        self.session = ort.InferenceSession(self.model)

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
        print(f"result[0] min {np.min(result[0])}")
        print(f"result[0] max {np.max(result[0])}")
        seg_result = result[0][0, :, :, 0]
        seg_result = np.clip(seg_result, 0 , 1)
        print(f"result[0] min {np.min(seg_result)}")
        print(f"result[0] max {np.max(seg_result)}")
        bin_mask = (seg_result > 0.05).astype(np.uint8)*255
        print(f"result[0] min {np.min(bin_mask)}")
        print(f"result[0] max {np.max(bin_mask)}")
        # result = result.squeeze().reshape(424, 424)
        
        # mask = (result > 0.5).astype(np.uint8)
        # mask = mask*255
        #mask_bin = cv.threshold(mask, 0.5, 1, cv.THRESH_BINARY)*255
        #mask_bin = np.reshape(mask_bin, (512, 424))
        # resized_img = cv.resize(thresholded_result, (512, 424))
        
        return bin_mask
    
    def predict(self, image):
        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        input_type = self.session.get_inputs()[0].type

        
        output_name = self.session.get_outputs()[0].name
        output_shape = self.session.get_outputs()[0].shape
        output_type = self.session.get_outputs()[0].type
        print(output_shape)
        print(output_type)
        ort_outs = self.session.run([output_name], {input_name : image })
        self.prediction = False
        return ort_outs
    
    def callback(self, msg):
        if not self.prediction:
            try:
                rgbd_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding = "passthrough")
            except CvBridgeError as e:
                print(e)
            #print(np.shape(msg))
            cv.imshow("test", rgbd_img)
            rgbd_img = rgbd_img/255
            rgbd_img = trans.resize(rgbd_img, (424,424))
            
            #cv.imshow("test", rgbd_img)
            rgbd_img = np.array(rgbd_img, dtype = np.float32)
            rgbd_img = np.expand_dims(rgbd_img, axis = 0)
            #rgbd_img = np.reshape(rgbd_img,(1,) +rgbd_img.shape).astype(np.float32)
            self.img_and_stamp = rgbd_img
            self.prediction = True
            
            if self.prediction:
                rospy.loginfo("Test image has been loaded.")
                rospy.loginfo("Prediction will start...")
                
                test_img= self.img_and_stamp
                print(np.shape(test_img))
                rospy.loginfo("Predicting the object...")
                result = self.predict(test_img)
                
                # print(result[1].shape)
                # print(result[2].shape)
                # print(result[3].shape)
                predicted_mask = self.postprocess_image(result)
                #print(predicted_mask)
                print(np.shape(predicted_mask))
                # bin_mask =cv.bitwise_and(test_img,redicted_mask)
                mask_msg = self.bridge.cv2_to_imgmsg(predicted_mask)
                
                self.publisher.publish(mask_msg)
                rospy.loginfo("Publishing the mask")
                plt.imshow(cv.cvtColor(predicted_mask,cv.COLOR_BGR2RGB))
                #cv.imshow("predicted mask", predicted_mask)
                self.prediction = False
                plt.show()


if __name__ == '__main__':
    rospy.init_node('semantic_segmentation_node')
    node = SemanticSegmentationNode()
    rospy.spin()
    

