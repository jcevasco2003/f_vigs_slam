#pragma once

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/approximate_time.hpp"
#include "message_filters/cache.hpp"
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

// Definimos la clase que se encarga de la logica del nodo en si mismo

namespace f_vigs_slam
{
    class GSSlam;
    class GaussianSplattingViewer;
    struct IntrinsicParameters;
    struct ImuData;
    struct CameraPose;
    class Preintegration;

    class GSSlamNode : public rclcpp::Node
    {
    public:
        explicit GSSlamNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
        ~GSSlamNode();

    protected:
        struct Impl;
        std::unique_ptr<Impl> impl_;

        bool hasIntrinsics = false;
        bool hasImu = false;
        bool imuInitialized = false;
        bool isProcessing = false;
        bool hasCameraInfo = false;
        cv::Mat rgbImg, depthImg;
        cv::Mat local_rgb_img, local_depth_img;
        int nb_init_imu_ = 0;
        Eigen::Vector3d avg_acc_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d avg_gyro_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d acc_bias_ = Eigen::Vector3d::Zero();
        Eigen::Quaterniond q_imu_cam_ = Eigen::Quaterniond::Identity();
        Eigen::Vector3d t_imu_cam_ = Eigen::Vector3d::Zero();
        double depth_scale_ = 0.001;
        int gauss_init_size_px_ = 7;
        double gauss_init_scale_ = 0.01;
        rclcpp::Time last_imu_stamp_{0, 0, RCL_ROS_TIME};
        rclcpp::Time last_processed_rgbd_stamp_{0, 0, RCL_ROS_TIME};
        rclcpp::Time last_rgb_stamp_{0, 0, RCL_ROS_TIME};
        sensor_msgs::msg::Image::ConstSharedPtr last_rgb_msg_;

        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
        std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
        std::string imu_frame_id_;
        std::string camera_frame_id_;

        // Suscripciones a los topicos de ros2
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
        std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_sub_;
        std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> color_sub_;
        using RGBDSyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
        std::shared_ptr<message_filters::Synchronizer<RGBDSyncPolicy>> rgbd_sync_;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
        message_filters::Cache<sensor_msgs::msg::Imu> imu_cache_preint_{1000};

        // Publisher de odometría
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_imu_pub_;
        
        // Mensaje y frame para odometría
        nav_msgs::msg::Odometry odom_msg_;
        nav_msgs::msg::Odometry odom_imu_msg_;
        std::string world_frame_id_;

        // Guardamos los tiempos de los ultimos callbacks recibidos
        rclcpp::Time last_imu_callback_time_{0, 0, RCL_CLOCK_UNINITIALIZED};
        rclcpp::Time last_rgbd_callback_time_{0, 0, RCL_CLOCK_UNINITIALIZED};
        rclcpp::Time last_color_diag_time_{0, 0, RCL_CLOCK_UNINITIALIZED};
        rclcpp::Time last_depth_diag_time_{0, 0, RCL_CLOCK_UNINITIALIZED};
        rclcpp::Time last_sync_process_time_{0, 0, RCL_ROS_TIME};
        
        // Diagnostic subscribers for monitoring individual message arrivals
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_diag_sub_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_diag_sub_;

        // Callbacks para manejo de mensajes
        void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
        void rgbdCallback(const std::shared_ptr<const sensor_msgs::msg::Image>& color,
                  const std::shared_ptr<const sensor_msgs::msg::Image>& depth);
        void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
        void imuPreintegratedCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
        void colorDiagCallback(const sensor_msgs::msg::Image::SharedPtr msg);
        void depthDiagCallback(const sensor_msgs::msg::Image::SharedPtr msg);
        
        // IMU processing helpers
        void syncRgbdImu();
        Eigen::Matrix3d computeGravityAlignment(const Eigen::Vector3d& acc) const;

        //
        void processCallbacks(); 
    };
} // namespace f_vigs_slam

