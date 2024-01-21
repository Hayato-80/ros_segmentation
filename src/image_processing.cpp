#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <iostream>
#include <iomanip>
#include <vector>

cv::Mat background_image_depth;
const double MAX_DEPTH = 5.0;
int image_count = 0;
int frame_count = 0;

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

//const sensor_msgs::PointCloud2ConstPtr& msg
cv::Mat convertPointCloud2ToCvImageDepth(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);
     // Extract depth values from the PointCloud2
     cv::Mat depth_image = cv::Mat::ones(cloud.height, cloud.width, CV_32FC1)*5.0;
     for (int i = 0; i < cloud.height; ++i) {
         for (int j = 0; j < cloud.width; ++j) {
             depth_image.at<float>(i, j) = std::sqrt(std::pow(cloud.at(j, i).x, 2.) + std::pow(cloud.at(j, i).y, 2.) + std::pow(cloud.at(j, i).z, 2.));
         }
     }

    return depth_image;
}

void obtainBackground(const cv::Mat& depth_image)
{
    if(depth_image.empty())
        return;

    if(background_image_depth.empty())
    {
        background_image_depth = depth_image.clone();
    }
    else
    {
        background_image_depth = cv::min(background_image_depth ,depth_image);
    }
}

cv::Mat subtractBackground(const cv::Mat& input_image)
{
    if(background_image_depth.empty())
        obtainBackground(input_image);
        //return;
    cv::Mat foreground_mask_depth;
    if(!background_image_depth.empty())
    {
        foreground_mask_depth = (background_image_depth - background_image_depth*0.0) - input_image;
        cv::inRange(foreground_mask_depth, cv::Scalar(0.0125), cv::Scalar(6.0), foreground_mask_depth);

        int morph_size = 1;
        // Element:\n 0/cv::MORPH_RECT: Rect , 1/cv::MORPH_CROSS: Cross,  2/cv::MORPH_ELLIPSE: Ellipse
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
            cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
            cv::Point(morph_size, morph_size));

        cv::Mat foreground_mask_depth_median;
        cv::medianBlur(foreground_mask_depth, foreground_mask_depth_median,5);
        foreground_mask_depth &= foreground_mask_depth_median;

        cv::morphologyEx(foreground_mask_depth, foreground_mask_depth, cv::MORPH_ERODE, element, cv::Point(-1, -1), 1);
        cv::morphologyEx(foreground_mask_depth, foreground_mask_depth, cv::MORPH_OPEN, element, cv::Point(-1, -1), 2);
    
        cv::Mat image_foreground_depth = cv::Mat::zeros(input_image.rows, input_image.cols, input_image.type());
        input_image.copyTo(image_foreground_depth, foreground_mask_depth);
        cv::normalize(image_foreground_depth, image_foreground_depth, 0, 255, cv::NORM_MINMAX);
        cv::threshold(image_foreground_depth, image_foreground_depth, 100, 255, cv::THRESH_BINARY);
        return image_foreground_depth;
    }
    return cv::Mat();
}

cv::Mat labeling(const cv::Mat& input_image){
    cv::Mat label_img;
    cv::Mat stats, centroids;
    cv::Mat image_cvt;
    input_image.convertTo(image_cvt, CV_8SC1, 1, 0);
    int nLabs = cv::connectedComponentsWithStats(image_cvt, label_img, stats, centroids, 8, CV_32S);
    std::vector<cv::Vec3b> colors(nLabs);
    colors[0] = cv::Vec3b(0, 0, 0); //background
    for (int i = 1; i < nLabs; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= 300 && area <= 520) {
            colors[i] = cv::Vec3b(0, 255, 0);
        } 
        // else if (area >= 1000) {
        //     colors[i] = cv::Vec3b(0, 0, 255);
        // }
    }

    cv::Mat new_img(input_image.size(), CV_8UC3);
    for (int y = 0; y < new_img.rows; ++y) {
        for (int x = 0; x < new_img.cols; ++x) {
            int label = label_img.at<int>(y, x);
            cv::Vec3b &pixel = new_img.at<cv::Vec3b>(y, x);
            pixel = colors[label];
        }
    }
    return new_img;
}

void saveImages(const cv::Mat& cv_image, const cv::Mat& depth_image,  const cv::Mat& label_image) {
    // making a rgbd image
    cv::normalize(depth_image, depth_image, 0, 2, cv::NORM_MINMAX);
    cv::Mat depth_image_8bit;
    depth_image.convertTo(depth_image_8bit, CV_8UC1);
    cv::Mat rgbd_image;
    cv::cvtColor(cv_image, rgbd_image, cv::COLOR_BGR2BGRA);  // Convert RGB to BGRA
    std::vector<cv::Mat> channels;
    cv::split(rgbd_image, channels);
    if (depth_image_8bit.size() != rgbd_image.size()) {
        ROS_ERROR("Depth image has a different size than RGB image");
        return;
    }
    else if (rgbd_image.depth() != depth_image_8bit.depth()) {
        ROS_ERROR("RGB and depth images have different depths");
        return;
    }
    channels[3] = depth_image_8bit;  // Replace alpha channel with the depth image
    cv::merge(channels, rgbd_image);

    // start saving from the frame that has an object
    if (frame_count>9){
        ROS_INFO("Saving images...");
        // set file name
        std::ostringstream rgbd_filename_png, label_filename;
        rgbd_filename_png << std::setw(4) << std::setfill('0') << image_count  << ".png";
        label_filename << std::setw(4) << std::setfill('0') << image_count << ".png";
        // save each image
        cv::imwrite("src/image_processing/datasets/images/" + rgbd_filename_png.str(), rgbd_image);
        cv::imwrite("src/image_processing/datasets/labels/" + label_filename.str(), label_image);
        image_count++;
        ROS_INFO("Total images saved: %d", image_count);
    }
    frame_count++;
}

void depthCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    // Convert PointCloud2 to depth image
    ROS_INFO("Image processing has started.");
    cv::Mat cv_image = convertPointCloud2ToRGB(msg);
    cv::Mat depth_image = convertPointCloud2ToCvImageDepth(msg);

    // Subtract background
    cv::Mat foreground = subtractBackground(depth_image);
    
    cv::Mat label = labeling(foreground);
    ROS_INFO("Image processing has finished.");
    ROS_INFO("The label has been created.");

    saveImages(cv_image, depth_image, label);
    
    // Display the depth image
    cv::imshow("cv_image", cv_image);
    cv::imshow("Depth", depth_image);
    cv::imshow("Foreground", foreground);
    cv::imshow("Label", label);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_processing");
    ros::NodeHandle nh;
    
    ros::Subscriber depth_sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 1000, depthCallback);

    ros::spin();

    return 0;
}