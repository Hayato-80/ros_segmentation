#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <boost/filesystem.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <iostream>
#include <filesystem>

cv::Mat rgbd_img;
ros::Publisher rgbd_pub;
cv::Mat convertPointCloud2ToRGB(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    sensor_msgs::Image image;
    cv_bridge::CvImagePtr cv_ptr;
    try{
        // convert cloud data
        pcl::toROSMsg(*msg, image);
    }
    catch(std::runtime_error e){
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    cv_ptr = cv_bridge::toCvCopy(image);
    return cv_ptr->image;
}

// const sensor_msgs::PointCloud2ConstPtr& msg
cv::Mat convertPointCloud2ToCvImageDepth(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    pcl::PointCloud <pcl::PointXYZ> cloud;
    //pcl::PointXYZ cloud;
    pcl::fromROSMsg(*msg, cloud);
    // Extract depth values from the PointCloud2
    cv::Mat depth_image = cv::Mat::ones(cloud.height, cloud.width, CV_32FC1)*5.0;
    for (int i = 0; i < cloud.height; ++i) {
        for (int j = 0; j < cloud.width; ++j) {
            depth_image.at<float>(i, j) = std::sqrt(std::pow(cloud.at(j, i).x, 2.) + std::pow(cloud.at(j, i).y, 2.) + std::pow(cloud.at(j, i).z, 2.));
        }
    }
    double max;
    cv::minMaxLoc(depth_image,nullptr,&max);
    double mean=cv::mean(depth_image)[0];


    std::cout<<max<<std::endl;
    //std::cout<<depth_image<<std::endl;
    return depth_image;
}

cv::Mat publish_rgbd(const cv::Mat& cv_image, const cv::Mat& depth_image) {
    // making a rgbd image
    cv::normalize(depth_image, depth_image, 0, 1, cv::NORM_MINMAX);
    cv::Mat depth_image_8bit;
    depth_image.convertTo(depth_image_8bit, CV_8UC1);
    cv::Mat rgbd_image;
    cv::cvtColor(cv_image, rgbd_image, cv::COLOR_BGR2RGBA);  // Convert RGB to BGRA
    std::vector<cv::Mat> channels;
    cv::split(rgbd_image, channels);
    // if (depth_image_8bit.size() != rgbd_image.size()) {
    //     ROS_ERROR("Depth image has a different size than RGB image");
    //     //return;
    // }
    // else if (rgbd_image.depth() != depth_image_8bit.depth()) {
    //     ROS_ERROR("RGB and depth images have different depths");
    //     //return;
    // }
    channels[3] = depth_image_8bit;  // Replace alpha channel with the depth image
    cv::merge(channels, rgbd_image);

    //rgbd_image = rgbd_image/255;
    return rgbd_image;
}

void ImageCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    ROS_INFO("PointCloud2 is subscribed.");
    cv::Mat cv_img = convertPointCloud2ToRGB(msg);
    
    cv::Mat depth_img = convertPointCloud2ToCvImageDepth(msg);

    rgbd_img = publish_rgbd(cv_img, depth_img);


    //cv_bridge::CvImage img_bridge;
    sensor_msgs::ImagePtr img_msg;
    
    img_msg = (cv_bridge::CvImage(std_msgs::Header(), "rgba8", rgbd_img)).toImageMsg();

    //img_bridge.toImageMsg(img_msg);
    img_msg->header.stamp = msg->header.stamp;
    img_msg->header.frame_id = msg->header.frame_id;
    ROS_INFO("Publishing the message...");
    rgbd_pub.publish(img_msg);
    // cv::imshow("RGBD", rgbd_img);
    // cv::waitKey();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "rgbd_publisher");
    ros::NodeHandle nh;
    ROS_INFO("Node has been started.");
    ros::Subscriber img_sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 1, ImageCallback);
    
    rgbd_pub = nh.advertise<sensor_msgs::Image>("/rgbd_image_topic",1);
    //ros::Rate loop_rate(30);
    // while(ros::ok()){
        
    //     rgbd_pub.publish(img_msg);
    //     loop_rate.sleep();
    // }
    ros::spin();

    return 0;
}