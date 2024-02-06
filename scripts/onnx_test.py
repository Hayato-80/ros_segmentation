#!/usr/bin/env python3

import onnxruntime as ort
import skimage.transform as trans
import sys 
sys.path.append('/home/kojima/catkin_ws/src/ros_segmentation')
import os
import time
import rospy
#import scripts.data

import cv2 as cv
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image, PointCloud2

Ball = [0, 255, 0]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Ball, Unlabelled])

class SemanticSegmentationNode:
    def __init__(self):
        self.model = 'src/ros_segmentation/model/unet_test/model_op11_1.onnx'
        self.session = ort.InferenceSession(self.model)
        #,providers=['CUDAExecutionProvider'])
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

    # def labelVisualize(self,num_class,color_dict,img):
    # img = img[:,:,0] if len(img.shape) == 3 else img
    # img_out = np.zeros(img.shape + (3,))
    # for i in range(num_class):
    #     img_out[img == i,:] = color_dict[i]
    # #return img_out
    # return img_out / 255
        
    # def load_trtengine(self, engine):
    #     print("Loading TRT engine from file :",engine)
    #     with open(engine, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    #         engine = runtime.deserialize_cuda_engine(f.read())
    #         return engine
    #def saveResult(self,npyfile,flag_multi_class = False,num_class = 2):
    # for i,item in enumerate(npyfile):
    #     img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
    #     print(img) ## ex. [[0.16509348 0.08227982 ... 0.07113015]
    #     img = (img > 0.5).astype(np.uint8) # .reshape(256, 256)
    #     print(img) ## ex. [[ 0 0 0 0 ... 1 1 1 ]]
    #     img = (img * 255)
    #     print(img) ## ex. [[0 0 00 ... 255 255 255]]
    #     return img
    #     #cv.imwrite(os.path.join(save_path,"%d_predict.png"%i), img)
        
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
        # print(f"result[0] min {np.min(result[0])}")
        # print(f"result[0] max {np.max(result[0])}")
        # pred_onnx =  np.array(result)
        #pred_onnx =  np.array(result)[1]*255
        #seg_result = np.clip(pred_onnx, 0 , 1)
        #print(np.shape(pred_onnx))
        # result = result.squeeze()
        pred_onnx= result[0][0,:, :,0]
        pred_onnx = (pred_onnx > 0.5).astype(np.uint8)*255

        #pred_onnx =pred_onnx * 255
        #print(pred_onnx)
        #seg_result = seg_result.reshape(424,424,1)
        # seg_result = np.reshape(seg_result)(424, 424)
        # seg_result = seg_result[0][0, :, :, 0]
        # seg_result = np.clip(seg_result, 0 , 1)
        # # print(f"result[0] min {np.min(seg_result)}")
        # # print(f"result[0] max {np.max(seg_result)}")
        # bin_mask = (seg_result > 0.3).astype(np.uint8)*255
        # print(f"result[0] min {np.min(bin_mask)}")
        # print(f"result[0] max {np.max(bin_mask)}")
        # 
        # bin_mask =  np.array(result)
        # bin_mask =  np.array(bin_mask).reshape(424,424)
        
        # bin_mask = (bin_mask > 0.5).astype(np.uint8)
        # bin_mask =bin_mask * 255

        # mask = (result > 0.5).astype(np.uint8)
        # mask = mask*255
        #mask_bin = cv.threshold(mask, 0.5, 1, cv.THRESH_BINARY)*255
        #mask_bin = np.reshape(mask_bin, (512, 424))
        # resized_img = cv.resize(thresholded_result, (512, 424))
        
        return pred_onnx
    
    def predict(self, image):
        start_time = time.time()
        input_name = self.session.get_inputs()[0].name
        
        #output_name = self.session.get_outputs()[0].name
        
        ort_outs = self.session.run(None, {input_name : image})

        print("Predict time:"+str(time.time() - start_time))
        self.prediction = False
        return ort_outs
    
    def callback(self, msg):
        if not self.prediction:
            try:
                #rgbd_img = cv.imread("test/segmentation_win/datasets/test/test2/0.png", cv.IMREAD_UNCHANGED)
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
            # plt.imshow(rgbd_img)
                #cv.imshow("predicted mask", predicted_mask)
            # plt.show()
            # cv.imshow("test", rgbd_img)
            cv.imshow("test", rgbd_img)
            #rgbd_img = np.array(rgbd_img, dtype = np.float32)
            #rgbd_img = np.expand_dims(rgbd_img, axis = 0)
            rgbd_img = np.reshape(rgbd_img,(1,) +rgbd_img.shape).astype(np.float32)
            self.img_and_stamp = rgbd_img
            self.prediction = True
            
            if self.prediction:
                rospy.loginfo("Test image has been loaded.")
                rospy.loginfo("Prediction will start...")
                
                test_img= self.img_and_stamp
                #print(np.shape(test_img))
                rospy.loginfo("Predicting the object...")
                result = self.predict(test_img)
                # print(np.shape(result))
                # print(result[1].shape)
                # print(result[2].shape)
                # print(result[3].shape)
                predicted_mask = self.postprocess_image(result)
                #print(predicted_mask)
                print("output shape",np.shape(predicted_mask))
               
                mask_msg = self.bridge.cv2_to_imgmsg(predicted_mask)
                
                self.publisher.publish(mask_msg)
                rospy.loginfo("Publishing the mask")
                #plt.imshow(predicted_mask,cmap='gray')
                cv.imshow("predicted mask", predicted_mask)
                cv.waitKey(5)

                #plt.show()
                self.prediction = False
                


if __name__ == '__main__':
    rospy.init_node('semantic_segmentation_node')
    node = SemanticSegmentationNode()
    rospy.spin()
    

