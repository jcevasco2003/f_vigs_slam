#include <f_vigs_slam/GSSlamNode.hpp>
#include "rclcpp/rclcpp.hpp"
#include <f_vigs_slam/GSSlam.cuh>
#include <f_vigs_slam/GaussianSplattingViewer.hpp>
#include <f_vigs_slam/RepresentationClasses.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <functional>
#include <string>
#include <limits>

// Implementamos toda la logica del nodo

namespace f_vigs_slam
{
    struct GSSlamNode::Impl
    {
        GSSlam gs_core_;
        std::shared_ptr<GaussianSplattingViewer> viewer_;
        IntrinsicParameters intrinsics;
        ImuData imu_data_;
        Preintegration preint_;
        CameraPose odom_pose_init_;
    };

    GSSlamNode::GSSlamNode(const rclcpp::NodeOptions & options)
        : Node("gs_slam_node", options),
          impl_(std::make_unique<Impl>())
        {
            RCLCPP_INFO(this->get_logger(), "GS_Node has been started.");

            tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
            tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

            // Declaramos parametros y topicos del nodo, luego los leemos
            this->declare_parameter<std::string>("imu_topic", "imu");
            this->declare_parameter<std::string>("depth_topic", "image_depth");
            this->declare_parameter<std::string>("color_topic", "image_color");
            this->declare_parameter<std::string>("camera_info_topic", "camera_info");
            this->declare_parameter<std::string>("imu_preint_topic", "");
            this->declare_parameter<std::string>("world_frame_id", "world");

            this->declare_parameter<double>("acc_n", 0.1);
            this->declare_parameter<double>("gyr_n", 0.01);
            this->declare_parameter<double>("acc_w", 0.001);
            this->declare_parameter<double>("gyr_w", 0.0001);

            auto imu_topic = this->get_parameter("imu_topic").as_string();
            auto depth_topic = this->get_parameter("depth_topic").as_string();
            auto color_topic = this->get_parameter("color_topic").as_string();
            auto camera_info_topic = this->get_parameter("camera_info_topic").as_string();
            auto imu_preint_topic = this->get_parameter("imu_preint_topic").as_string();
            if (imu_preint_topic.empty()) {
                imu_preint_topic = imu_topic;
                RCLCPP_INFO(this->get_logger(), "Using imu_topic '%s' for IMU subscription", imu_preint_topic.c_str());
            }
            world_frame_id_ = this->get_parameter("world_frame_id").as_string();

            this->get_parameter<double>("acc_n", impl_->imu_data_.acc_n);
            this->get_parameter<double>("gyr_n", impl_->imu_data_.gyr_n);
            this->get_parameter<double>("acc_w", impl_->imu_data_.acc_w);
            this->get_parameter<double>("gyr_w", impl_->imu_data_.gyr_w);

            this->declare_parameter<int>("gauss_init_size_px", 7);
            this->declare_parameter<double>("gauss_init_scale", 0.01);
            this->declare_parameter<double>("depth_scale", 0.001);
            this->declare_parameter<int>("pose_iterations", 4);
            this->declare_parameter<int>("gaussian_iterations", 10);
            this->declare_parameter<double>("eta_pose", 0.01);
            this->declare_parameter<double>("eta_gaussian", 0.002);
            this->declare_parameter<std::string>("gaussian_sampling_method", "beta_binomial");

            gauss_init_size_px_ = this->get_parameter("gauss_init_size_px").as_int();
            gauss_init_scale_ = this->get_parameter("gauss_init_scale").as_double();
            depth_scale_ = this->get_parameter("depth_scale").as_double();

            impl_->gs_core_.setGaussInitSizePx(gauss_init_size_px_);
            impl_->gs_core_.setGaussInitScale(static_cast<float>(gauss_init_scale_));
            
            // Configurar parámetros de optimización
            impl_->gs_core_.setPoseIterations(this->get_parameter("pose_iterations").as_int());
            impl_->gs_core_.setGaussianIterations(this->get_parameter("gaussian_iterations").as_int());
            impl_->gs_core_.setEtaPose(static_cast<float>(this->get_parameter("eta_pose").as_double()));
            impl_->gs_core_.setEtaGaussian(static_cast<float>(this->get_parameter("eta_gaussian").as_double()));
            impl_->gs_core_.setGaussianSamplingMethod(this->get_parameter("gaussian_sampling_method").as_string());

            // Subscribimos los nodos a los topicos
            imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
                imu_preint_topic, rclcpp::SensorDataQoS(),
                std::bind(&GSSlamNode::imuCallback, this, std::placeholders::_1));

            auto sensor_qos = rclcpp::SensorDataQoS();
            depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
                this, depth_topic, sensor_qos);
            color_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
                this, color_topic, sensor_qos);

            // Para el algoritmo es necesario que las imagenes de color y profundidad
            // esten sincronizadas
            rgbd_sync_ = std::make_shared<message_filters::Synchronizer<RGBDSyncPolicy>>(
                RGBDSyncPolicy(10), *color_sub_, *depth_sub_);
            rgbd_sync_->registerCallback(
                std::bind(&GSSlamNode::rgbdCallback, this, std::placeholders::_1, std::placeholders::_2));

            camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
                camera_info_topic, rclcpp::SensorDataQoS(),
                std::bind(&GSSlamNode::cameraInfoCallback, this, std::placeholders::_1));

            // TESTING: Diagnosticos para verificar que los mensajes llegan correctamente a los callbacks
            color_diag_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                color_topic, rclcpp::SensorDataQoS(),
                std::bind(&GSSlamNode::colorDiagCallback, this, std::placeholders::_1));
            depth_diag_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                depth_topic, rclcpp::SensorDataQoS(),
                std::bind(&GSSlamNode::depthDiagCallback, this, std::placeholders::_1));

            // Crear publishers de odometría
            odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
            odom_imu_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom_imu", 10);
            
            // Configurar frame_id de mensajes de odometría
            odom_msg_.header.frame_id = world_frame_id_;
            odom_imu_msg_.header.frame_id = world_frame_id_;

            RCLCPP_INFO(this->get_logger(), "Odometry publishers created on topics 'odom' and 'odom_imu'");

            this->declare_parameter<bool>("viewer", true);
            if (this->get_parameter("viewer").as_bool())
            {
                impl_->viewer_ = std::make_shared<GaussianSplattingViewer>(impl_->gs_core_);
                impl_->viewer_->startThread();
            }


            
        }

    GSSlamNode::~GSSlamNode()
    {
    }

    // Callbacks
    Eigen::Matrix3d GSSlamNode::computeGravityAlignment(const Eigen::Vector3d& acc) const
    {
        if (acc.norm() < 1e-6) {
            return Eigen::Matrix3d::Identity();
        }
        Eigen::Vector3d acc_norm = acc.normalized();
        Eigen::Vector3d gravity(0.0, 0.0, 1.0);
        Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(gravity, acc_norm);
        return q.toRotationMatrix();
    }

    void GSSlamNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "CameraInfo received! frame_id=%s", msg->header.frame_id.c_str());
        RCLCPP_INFO(this->get_logger(), "imu_frame_id_='%s', camera_frame_id_='%s'", 
                    imu_frame_id_.c_str(), camera_frame_id_.c_str());
        
        // Los parametros intrinsecos se representan en K con K =:
        // [fx  0 cx]
        // [ 0 fy cy]
        // [ 0  0  1]
        impl_->intrinsics.f = float2{static_cast<float>(msg->k[0]),
                      static_cast<float>(msg->k[4])};
        impl_->intrinsics.c = float2{static_cast<float>(msg->k[2]),
                      static_cast<float>(msg->k[5])};
        camera_frame_id_ = msg->header.frame_id;

        impl_->gs_core_.setIntrinsics(impl_->intrinsics);
        hasIntrinsics = true;

        if (!imu_frame_id_.empty() && !camera_frame_id_.empty())
        {
            RCLCPP_INFO(this->get_logger(), "Looking up transform from %s to %s", 
                        imu_frame_id_.c_str(), camera_frame_id_.c_str());
            try
            {
                // Use current timestamp with 0.5s timeout for TF lookup
                auto t = tf_buffer_->lookupTransform(
                    imu_frame_id_, camera_frame_id_,
                    this->now(), tf2::Duration(std::chrono::milliseconds(500)));

                tf2::fromMsg(t.transform.rotation, q_imu_cam_);    // rotate from camera to IMU
                tf2::fromMsg(t.transform.translation, t_imu_cam_); // pos camera in imu frame
                impl_->gs_core_.setImuToCamExtrinsics(t_imu_cam_, q_imu_cam_);
                hasCameraInfo = true;
                RCLCPP_INFO(this->get_logger(), "Transform found! hasCameraInfo=true");
            }
            catch (const tf2::TransformException &ex)
            {
                RCLCPP_WARN(this->get_logger(), "Transform lookup failed: %s", ex.what());
                RCLCPP_WARN(this->get_logger(), "No IMU->camera transform yet: %s", ex.what());
                RCLCPP_WARN(this->get_logger(), "Ensure TF tree connects: world -> ... -> %s and world -> ... -> %s", 
                            imu_frame_id_.c_str(), camera_frame_id_.c_str());
            }
        }
    }

    void GSSlamNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        last_imu_callback_time_ = this->now();
        imu_frame_id_ = msg->header.frame_id;

        static int imu_count = 0;
        if (++imu_count % 50 == 0) {
            RCLCPP_INFO(this->get_logger(), "IMU received: count=%d, frame_id=%s, hasCameraInfo=%d, nb_init_imu=%d",
                imu_count, msg->header.frame_id.c_str(), hasCameraInfo, nb_init_imu_);
        }

        Eigen::Vector3d raw_acc(msg->linear_acceleration.x,
                                msg->linear_acceleration.y,
                                msg->linear_acceleration.z);

        Eigen::Vector3d raw_gyro(msg->angular_velocity.x,
                                 msg->angular_velocity.y,
                                 msg->angular_velocity.z);

        // Intentar obtener transform si aún no está listo
        if (!hasCameraInfo && hasIntrinsics && !camera_frame_id_.empty())
        {
            try
            {
                // Use current timestamp with timeout for TF lookup
                auto t = tf_buffer_->lookupTransform(
                    imu_frame_id_, camera_frame_id_,
                    this->now(), tf2::Duration(std::chrono::milliseconds(100)));

                tf2::fromMsg(t.transform.rotation, q_imu_cam_);
                tf2::fromMsg(t.transform.translation, t_imu_cam_);
                impl_->gs_core_.setImuToCamExtrinsics(t_imu_cam_, q_imu_cam_);
                hasCameraInfo = true;
            }
            catch (const tf2::TransformException &ex)
            {
                RCLCPP_DEBUG(this->get_logger(), "No IMU->camera transform yet: %s", ex.what());
            }
        }

        // Sin cámara no seguimos para mantener sincronía con RGBD
        if (!hasCameraInfo) {
            RCLCPP_INFO_ONCE(this->get_logger(), "Waiting for camera info before processing IMU");
            return;
        }

        // Acumulamos las primeras 100 muestras para estimar bias
        if (nb_init_imu_ < 100) {
            avg_acc_ += raw_acc;
            avg_gyro_ += raw_gyro;
            ++nb_init_imu_;
            last_imu_stamp_ = rclcpp::Time(msg->header.stamp);
            return;
        }

        rclcpp::Time stamp(msg->header.stamp);
        if (last_imu_stamp_.nanoseconds() == 0) {
            last_imu_stamp_ = stamp;
        }
        double dt = (stamp - last_imu_stamp_).seconds();
        last_imu_stamp_ = stamp;

        // Actualizar datos IMU actuales
        impl_->imu_data_.Acc = raw_acc;
        impl_->imu_data_.Gyro = raw_gyro;
        hasImu = true;

        // Rama de inicialización: calcular bias a partir de 100 muestras
        if (!imuInitialized) {
            avg_acc_ /= static_cast<double>(nb_init_imu_);
            avg_gyro_ /= static_cast<double>(nb_init_imu_);
            
            Eigen::Matrix3d R0 = computeGravityAlignment(avg_acc_);
            Eigen::Vector3d gravity(0.0, 0.0, 9.81);
            acc_bias_ = avg_acc_ - R0.inverse() * gravity;

            Eigen::Quaterniond q_init_imu(R0);
            Eigen::Quaterniond q_init_cam = q_init_imu * q_imu_cam_;

            // Inicializar pose para el primer frame
            impl_->odom_pose_init_.position.x = static_cast<float>(t_imu_cam_.x());
            impl_->odom_pose_init_.position.y = static_cast<float>(t_imu_cam_.y());
            impl_->odom_pose_init_.position.z = static_cast<float>(t_imu_cam_.z());
            // Guardamos en orden estándar (x, y, z, w)
            impl_->odom_pose_init_.orientation.x = static_cast<float>(q_init_cam.x());
            impl_->odom_pose_init_.orientation.y = static_cast<float>(q_init_cam.y());
            impl_->odom_pose_init_.orientation.z = static_cast<float>(q_init_cam.z());
            impl_->odom_pose_init_.orientation.w = static_cast<float>(q_init_cam.w());

            // Configurar imu_data_ completo con medición inicial
            impl_->imu_data_.Acc = avg_acc_;
            impl_->imu_data_.Gyro = avg_gyro_;
            impl_->imu_data_.g_norm = 9.81;
            
            // Inicializar preintegración en GSSlam y en el nodo
            impl_->gs_core_.initializeImu(impl_->imu_data_);
            impl_->preint_.init(impl_->imu_data_.Acc, impl_->imu_data_.Gyro, acc_bias_, avg_gyro_,
                         impl_->imu_data_.acc_n, impl_->imu_data_.gyr_n, impl_->imu_data_.acc_w, impl_->imu_data_.gyr_w);
            
            imuInitialized = true;
            RCLCPP_INFO(this->get_logger(), "IMU initialized:");
            RCLCPP_INFO(this->get_logger(), "  avg_acc = [%.4f %.4f %.4f] m/s²",
                        avg_acc_.x(), avg_acc_.y(), avg_acc_.z());
            RCLCPP_INFO(this->get_logger(), "  bias_acc = [%.4f %.4f %.4f] m/s²",
                        acc_bias_.x(), acc_bias_.y(), acc_bias_.z());
            RCLCPP_INFO(this->get_logger(), "  avg_gyro = [%.4f %.4f %.4f] rad/s",
                        avg_gyro_.x(), avg_gyro_.y(), avg_gyro_.z());
            RCLCPP_INFO(this->get_logger(), "  Preintegration initialized in GSSlam core");
            return;
        }

        if (dt > 1e-6) {
            impl_->preint_.add_imu(dt, impl_->imu_data_.Acc, impl_->imu_data_.Gyro);
        }

        const double *pose = impl_->gs_core_.getImuPose();
        const double *velocity = impl_->gs_core_.getImuVelocity();
        if (pose && velocity)
        {
            Eigen::Map<const Eigen::Vector3d> P(pose);
            Eigen::Quaterniond Q(pose[6], pose[3], pose[4], pose[5]);
            Eigen::Map<const Eigen::Vector3d> V(velocity);

            Eigen::Vector3d pos_imu, vel_imu;
            Eigen::Quaterniond rot_imu;
            impl_->preint_.predict(P, Q, V, pos_imu, rot_imu, vel_imu);

            odom_imu_msg_.header.stamp = msg->header.stamp;
            odom_imu_msg_.child_frame_id = msg->header.frame_id;
            odom_imu_msg_.pose.pose.position.x = pos_imu.x();
            odom_imu_msg_.pose.pose.position.y = pos_imu.y();
            odom_imu_msg_.pose.pose.position.z = pos_imu.z();
            odom_imu_msg_.pose.pose.orientation.x = rot_imu.x();
            odom_imu_msg_.pose.pose.orientation.y = rot_imu.y();
            odom_imu_msg_.pose.pose.orientation.z = rot_imu.z();
            odom_imu_msg_.pose.pose.orientation.w = rot_imu.w();
            odom_imu_msg_.twist.twist.linear.x = vel_imu.x();
            odom_imu_msg_.twist.twist.linear.y = vel_imu.y();
            odom_imu_msg_.twist.twist.linear.z = vel_imu.z();

            odom_imu_pub_->publish(odom_imu_msg_);
        }

        sensor_msgs::msg::Imu::SharedPtr imu_cache(new sensor_msgs::msg::Imu(*msg));
        imu_cache_preint_.add(imu_cache);
        RCLCPP_DEBUG(this->get_logger(), "IMU added to cache");

        syncRgbdImu();
    }

    void GSSlamNode::imuPreintegratedCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        RCLCPP_DEBUG(this->get_logger(), "IMU preintegrated received: frame_id=%s ts=%u.%u",
                     msg->header.frame_id.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);
    }

    void GSSlamNode::colorDiagCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        auto now = this->now();
        if (last_color_diag_time_.nanoseconds() == 0 || (now - last_color_diag_time_).seconds() > 1.0) {
            RCLCPP_INFO(this->get_logger(), "[DIAG] Color image received: %ux%u, encoding=%s, stamp=%ld.%09ld",
                msg->width, msg->height, msg->encoding.c_str(),
                msg->header.stamp.sec, msg->header.stamp.nanosec);
            last_color_diag_time_ = now;
        }
    }

    void GSSlamNode::depthDiagCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        auto now = this->now();
        if (last_depth_diag_time_.nanoseconds() == 0 || (now - last_depth_diag_time_).seconds() > 1.0) {
            RCLCPP_INFO(this->get_logger(), "[DIAG] Depth image received: %ux%u, encoding=%s, stamp=%ld.%09ld",
                msg->width, msg->height, msg->encoding.c_str(),
                msg->header.stamp.sec, msg->header.stamp.nanosec);
            last_depth_diag_time_ = now;
        }
    }

    void GSSlamNode::rgbdCallback(const std::shared_ptr<const sensor_msgs::msg::Image>& color,
                                  const std::shared_ptr<const sensor_msgs::msg::Image>& depth)
    {
        RCLCPP_INFO(this->get_logger(), "rgbdCallback called! hasCameraInfo=%d", hasCameraInfo);
        
        if (!hasCameraInfo) {
            RCLCPP_INFO(this->get_logger(), "rgbdCallback: hasCameraInfo=false, returning");
            return;
        }

        // Guardamos tiempos de llegada
        last_rgbd_callback_time_ = this->now();
        last_rgb_stamp_ = rclcpp::Time(color->header.stamp);
        last_rgb_msg_ = color;

        RCLCPP_INFO(this->get_logger(), "RGBD received: color %ux%u | depth %ux%u, stamp=%ld",
            color->width, color->height,
            depth->width, depth->height,
            last_rgb_stamp_.nanoseconds());

        
        // TODO: pasar ambas imágenes sincronizadas al núcleo de SLAM
        // Example: gs_core_.processRgbd(color->header.stamp, color, depth);

        // Guardamos las imagenes recibidas
        try
        {
            depthImg = cv_bridge::toCvShare(depth)->image;
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(),
                         "could not convert depth image with encoding '%s'", depth->encoding.c_str());
            return;
        }
        try
        {
            rgbImg = cv_bridge::toCvShare(color, "bgr8")->image;
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(),
                         "could not convert color image with encoding '%s'.", color->encoding.c_str());
                return;
        }

        local_depth_img = depthImg.clone();
        local_rgb_img = rgbImg.clone();

        syncRgbdImu();
    }

    void GSSlamNode::syncRgbdImu()
    {
        RCLCPP_INFO(this->get_logger(), "syncRgbdImu called");
        
        if (isProcessing) {
            RCLCPP_INFO(this->get_logger(), "syncRgbdImu: isProcessing=true");
            return;
        }
        if (!imuInitialized || !hasCameraInfo) {
            RCLCPP_INFO(this->get_logger(), "syncRgbdImu: imuInitialized=%d, hasCameraInfo=%d", imuInitialized, hasCameraInfo);
            return;
        }
        if (!impl_->gs_core_.getIsInitialized()) {
            RCLCPP_INFO(this->get_logger(), "syncRgbdImu: GSCore not initialized");
            return;
        }
        if (last_rgb_stamp_.nanoseconds() == 0) {
            RCLCPP_INFO(this->get_logger(), "syncRgbdImu: last_rgb_stamp=0");
            return;
        }

        // Use strict < comparison to allow frames with same timestamp (can happen with batching)
        // But warn about out-of-order delivery
        if (last_rgb_stamp_ < last_processed_rgbd_stamp_) {
            static rclcpp::Time last_warning = rclcpp::Time(0, 0, RCL_ROS_TIME);
            auto now = this->now();
            if ((now - last_warning).seconds() > 5.0) {
                RCLCPP_WARN(this->get_logger(), "Out-of-order RGBD frames detected: current(%.3f) < last(%.3f)",
                            last_rgb_stamp_.seconds(), last_processed_rgbd_stamp_.seconds());
                last_warning = now;
            }
            return;
        }

        rclcpp::Time last_imu_time(imu_cache_preint_.getLatestTime(), RCL_ROS_TIME);
        if (last_imu_time.nanoseconds() == 0) {
            RCLCPP_INFO(this->get_logger(), "syncRgbdImu: no IMU data in cache");
            return;
        }
        if (last_imu_time < last_rgb_stamp_) {
            RCLCPP_INFO(this->get_logger(), "syncRgbdImu: IMU data too old (last_imu_time < last_rgb_stamp)");
            return;
        }

        auto imu_interval = imu_cache_preint_.getInterval(last_processed_rgbd_stamp_, last_rgb_stamp_);
        if (imu_interval.empty()) {
            RCLCPP_INFO(this->get_logger(), "syncRgbdImu: empty IMU interval");
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "syncRgbdImu: processing %zu IMU measurements", imu_interval.size());

        rclcpp::Time t_prev = last_processed_rgbd_stamp_.nanoseconds() == 0
            ? rclcpp::Time(imu_interval.front()->header.stamp, RCL_ROS_TIME)
            : last_processed_rgbd_stamp_;

        for (const auto &m : imu_interval)
        {
            rclcpp::Time t_cur(m->header.stamp, RCL_ROS_TIME);
            double dt = (t_cur - t_prev).seconds();
            if (dt > 1e-6)
            {
                Eigen::Vector3d acc(m->linear_acceleration.x,
                                    m->linear_acceleration.y,
                                    m->linear_acceleration.z);
                Eigen::Vector3d gyr(m->angular_velocity.x,
                                    m->angular_velocity.y,
                                    m->angular_velocity.z);
                impl_->gs_core_.addImuMeasurement(dt, acc, gyr);
            }
            t_prev = t_cur;
        }

        last_processed_rgbd_stamp_ = last_rgb_stamp_;

        isProcessing = true;

        try
        {
            impl_->gs_core_.compute(local_rgb_img, local_depth_img, impl_->odom_pose_init_);
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in compute(): %s", e.what());
            isProcessing = false;
            return;
        }

        const double *pose = impl_->gs_core_.getImuPose();
        const double *velocity = impl_->gs_core_.getImuVelocity();

        odom_msg_.header.stamp = last_rgb_stamp_;
        odom_msg_.child_frame_id = last_rgb_msg_ ? last_rgb_msg_->header.frame_id : "camera_link";

        if (pose && velocity)
        {
            odom_msg_.pose.pose.position.x = pose[0];
            odom_msg_.pose.pose.position.y = pose[1];
            odom_msg_.pose.pose.position.z = pose[2];

            odom_msg_.pose.pose.orientation.x = pose[3];
            odom_msg_.pose.pose.orientation.y = pose[4];
            odom_msg_.pose.pose.orientation.z = pose[5];
            odom_msg_.pose.pose.orientation.w = pose[6];

            odom_msg_.twist.twist.linear.x = velocity[0];
            odom_msg_.twist.twist.linear.y = velocity[1];
            odom_msg_.twist.twist.linear.z = velocity[2];

            odom_msg_.twist.twist.angular.x = velocity[3];
            odom_msg_.twist.twist.angular.y = velocity[4];
            odom_msg_.twist.twist.angular.z = velocity[5];
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "getImuPose() returned null pointer!");
        }

        odom_pub_->publish(odom_msg_);
        
        RCLCPP_INFO(this->get_logger(), "Published odometry: pos=[%.3f, %.3f, %.3f]",
                    odom_msg_.pose.pose.position.x,
                    odom_msg_.pose.pose.position.y,
                    odom_msg_.pose.pose.position.z);

        // Publicar transformación TF correspondiente
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = odom_msg_.header.stamp;
        transform.header.frame_id = odom_msg_.header.frame_id;
        transform.child_frame_id = odom_msg_.child_frame_id;
        transform.transform.translation.x = odom_msg_.pose.pose.position.x;
        transform.transform.translation.y = odom_msg_.pose.pose.position.y;
        transform.transform.translation.z = odom_msg_.pose.pose.position.z;
        transform.transform.rotation = odom_msg_.pose.pose.orientation;
        tf_broadcaster_->sendTransform(transform);

        auto remaining_imu = imu_cache_preint_.getInterval(last_processed_rgbd_stamp_, imu_cache_preint_.getLatestTime());
        if (!remaining_imu.empty())
        {
            auto first = remaining_imu.front();
            Eigen::Vector3d acc(first->linear_acceleration.x,
                                first->linear_acceleration.y,
                                first->linear_acceleration.z);
            Eigen::Vector3d gyr(first->angular_velocity.x,
                                first->angular_velocity.y,
                                first->angular_velocity.z);

            Eigen::Vector3d ba = Eigen::Vector3d::Zero();
            Eigen::Vector3d bg = Eigen::Vector3d::Zero();
            if (velocity)
            {
                ba = Eigen::Vector3d(velocity[3], velocity[4], velocity[5]);
                bg = Eigen::Vector3d(velocity[6], velocity[7], velocity[8]);
            }

            rclcpp::Time t_imu(first->header.stamp, RCL_ROS_TIME);
            impl_->preint_.init(acc, gyr, ba, bg,
                         impl_->imu_data_.acc_n, impl_->imu_data_.gyr_n, impl_->imu_data_.acc_w, impl_->imu_data_.gyr_w);

            for (const auto &m : remaining_imu)
            {
                Eigen::Vector3d acc_i(m->linear_acceleration.x,
                                      m->linear_acceleration.y,
                                      m->linear_acceleration.z);
                Eigen::Vector3d gyr_i(m->angular_velocity.x,
                                      m->angular_velocity.y,
                                      m->angular_velocity.z);
                impl_->preint_.add_imu((rclcpp::Time(m->header.stamp, RCL_ROS_TIME) - t_imu).seconds(),
                                acc_i, gyr_i);
                t_imu = rclcpp::Time(m->header.stamp, RCL_ROS_TIME);
            }
        }

        isProcessing = false;
    }

    void GSSlamNode::processCallbacks(){
        if (!hasIntrinsics || !hasImu) {
            RCLCPP_WARN(this->get_logger(), "Waiting for intrinsics and IMU data before processing.");
            return;
        }

        if (isProcessing) return;

        isProcessing = true;

        // Procesar frame RGBD con el core del SLAM
        try {
            impl_->gs_core_.compute(local_rgb_img, local_depth_img, CameraPose());
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error in compute(): %s", e.what());
            isProcessing = false;
            return;
        }

        // TODO: integrar imu_buffer_ con las imágenes sincronizadas
        // - Procesar preintegración usando dt de cada muestra IMU
        // - Fusionar con pose visual del gs_core_

        // Publicar odometría después de procesar frame
        const double *pose = impl_->gs_core_.getImuPose();
        const double *velocity = impl_->gs_core_.getImuVelocity();
        
        if (pose && velocity)
        {
            odom_msg_.header.stamp = this->now();
            odom_msg_.child_frame_id = "camera_link";
            
            // Posición (P_cur[0-2])
            odom_msg_.pose.pose.position.x = pose[0];
            odom_msg_.pose.pose.position.y = pose[1];
            odom_msg_.pose.pose.position.z = pose[2];
            
            // Orientación (P_cur[3-6] = quaternion)
            odom_msg_.pose.pose.orientation.x = pose[3];
            odom_msg_.pose.pose.orientation.y = pose[4];
            odom_msg_.pose.pose.orientation.z = pose[5];
            odom_msg_.pose.pose.orientation.w = pose[6];
            
            // Velocidad lineal (VB_cur[0-2])
            odom_msg_.twist.twist.linear.x = velocity[0];
            odom_msg_.twist.twist.linear.y = velocity[1];
            odom_msg_.twist.twist.linear.z = velocity[2];
            
            // Velocidad angular (VB_cur[3-5] son biases acc, no velocidad angular)
            // En VIGS-Fusion usan indices 3-5 pero revisar si corresponde
            odom_msg_.twist.twist.angular.x = velocity[3];
            odom_msg_.twist.twist.angular.y = velocity[4];
            odom_msg_.twist.twist.angular.z = velocity[5];
            
            odom_pub_->publish(odom_msg_);
            
            // Publicar transformación TF correspondiente
            geometry_msgs::msg::TransformStamped transform;
            transform.header.stamp = odom_msg_.header.stamp;
            transform.header.frame_id = odom_msg_.header.frame_id;
            transform.child_frame_id = odom_msg_.child_frame_id;
            transform.transform.translation.x = odom_msg_.pose.pose.position.x;
            transform.transform.translation.y = odom_msg_.pose.pose.position.y;
            transform.transform.translation.z = odom_msg_.pose.pose.position.z;
            transform.transform.rotation = odom_msg_.pose.pose.orientation;
            tf_broadcaster_->sendTransform(transform);
            
            RCLCPP_DEBUG(this->get_logger(), 
                        "Published odometry: pos=[%.3f, %.3f, %.3f] vel=[%.3f, %.3f, %.3f]",
                        pose[0], pose[1], pose[2],
                        velocity[0], velocity[1], velocity[2]);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "getImuPose() returned null pointer!");
        }

        isProcessing = false;
    }



}