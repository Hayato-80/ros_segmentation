#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge
from keras.models import load_model
from sensor_msgs.msg import Image
import rosbag

class SemanticSegmentationNode:
    def __init__(self, bag_file):
        rospy.init_node('semantic_segmentation_node')
        self.model = load_model('/home/kojima/catkin_ws/src/image_processing/model/unet_class1_v03.hdf5')
        self.bridge = CvBridge()
        self.publisher = rospy.Publisher('/segmentation_result_topic', Image, queue_size=1)
        self.process_bag(bag_file)
        
    def convert_point_cloud_to_rgb(self, msg):
        image = PointCloud2()  # Assuming PointCloud2 is equivalent to sensor_msgs::Image
        cv_ptr = CvBridge().imgmsg_to_cv2(image)
        return cv_ptr

    def convert_point_cloud_to_depth_image(self, msg):
        cloud = pcl.PointCloud()
        pcl.fromROSMsg(msg, cloud)

        # Extract depth values from PointCloud2
        depth_image = np.ones((cloud.height, cloud.width), dtype=np.float32) * 5.0
        for i in range(cloud.height):
            for j in range(cloud.width):
                depth_image[i, j] = np.sqrt(cloud[j, i].x ** 2 + cloud[j, i].y ** 2 + cloud[j, i].z ** 2)

        return depth_image
    
    def combine_rgb_and_depth(self, rgb_image, depth_image):
        cv2.normalize(depth_image, depth_image, 0, 2, cv2.NORM_MINMAX)
        depth_image_8bit = depth_image.astype(np.uint8)
        rgbd_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2BGRA)
        channels = cv2.split(rgbd_image)

        # Ensure the depth image has the same size as the RGB image
        if depth_image_8bit.shape != rgbd_image.shape[:2]:
            rospy.logerror("Depth image has a different size than RGB image")
            return None

        # Ensure the depth image and RGB image have the same depth
        if rgbd_image.dtype != depth_image_8bit.dtype:
            rospy.logerror("RGB and depth images have different depths")
            return None

        # Replace the alpha channel with the depth image
        channels[3] = depth_image_8bit

        # Merge the channels back into an RGDB image
        rgbd_image = cv2.merge(channels)

        return rgbd_image
    
    def segment_image(self, image):
        desired_width = 424
        desired_height = 424
        # Resize the image to match the input size expected by your model
        resized_image = cv2.resize(image, (desired_width, desired_height))

        # Preprocess the image (normalize, etc.) according to your model's requirements
        # preprocessed_image = self.preprocess_image(resized_image)

        # Perform segmentation using your Keras model
        segmentation_result = self.model.predict(resized_image)

        # Post-process the segmentation result as needed
        postprocessed_result = self.postprocess_result(segmentation_result)

        return postprocessed_result

    # def preprocess_image(self, image):
    #     # Implement any preprocessing steps needed for your model
    #     # This may include normalization, resizing, etc.
    #     # Example: normalization to the range [0, 1]
    #     normalized_image = image.astype(np.float32) / 255.0

    #     return normalized_image

    def postprocess_result(self, segmentation_result):
        # Implement any postprocessing steps needed for your segmentation result
        # This may include thresholding, color mapping, etc.
        # Example: thresholding the result
        thresholded_result = (segmentation_result > 0.4).astype(np.uint8) * 255

        return thresholded_result

    def callback(self, point_cloud_msg):
        rgb_image = self.convert_point_cloud_to_rgb(point_cloud_msg)
        depth_image = self.convert_point_cloud_to_depth_image(point_cloud_msg)

        # Combine RGB and depth information or use one of them based on your requirements
        rgbd_image = self.combine_rgb_and_depth(rgb_image, depth_image)
        
        if rgbd_image is not None:
            segmented_image = self.segment_image(rgbd_image)
            # Publish the segmentation result
            segmentation_msg = CvBridge().cv2_to_imgmsg(segmented_image, encoding='bgra8')
            self.publisher.publish(segmentation_msg)

    def process_bag(self, bag_file):
        bag = rosbag.Bag(bag_file)
        for topic, msg, t in bag.read_messages(topics=['/camera/depth_registered/points']):
            self.callback(msg)
        bag.close()

if __name__ == '__main__':
    bag_file_path = '/home/kojima/ros_bags/rlm_rosbag_2023_11_07-13_21_16_2023_11_07-16_32_38.bag'
    node = SemanticSegmentationNode(bag_file_path)