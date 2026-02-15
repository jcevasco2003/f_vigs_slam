#include <f_vigs_slam/GSSlamNode.hpp>
#include "rclcpp/rclcpp.hpp"
#include <f_vigs_slam/RepresentationClasses.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <functional>
#include <string>
#include <limits>

// Implementamos toda la logica del nodo

namespace f_vigs_slam
{
    GSSlamNode::GSSlamNode(const rclcpp::NodeOptions & options)
        : Node("gs_slam_node", options)
        {
            RCLCPP_INFO(this->get_logger(), "GS_Node has been started.");

            tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

            // Declaramos parametros y topicos del nodo, luego los leemos
            this->declare_parameter<std::string>("imu_topic", "imu");
            this->declare_parameter<std::string>("depth_topic", "image_depth");
            this->declare_parameter<std::string>("color_topic", "image_color");
            this->declare_parameter<std::string>("camera_info_topic", "camera_info");
            this->declare_parameter<std::string>("imu_preint_topic", "imu_preint");
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
            }
            world_frame_id_ = this->get_parameter("world_frame_id").as_string();

            this->get_parameter<double>("acc_n", imu_data_.acc_n);
            this->get_parameter<double>("gyr_n", imu_data_.gyr_n);
            this->get_parameter<double>("acc_w", imu_data_.acc_w);
            this->get_parameter<double>("gyr_w", imu_data_.gyr_w);

            this->declare_parameter<int>("gauss_init_size_px", 7);
            this->declare_parameter<double>("gauss_init_scale", 0.01);
            this->declare_parameter<double>("depth_scale", 0.001);
            this->declare_parameter<int>("pose_iterations", 4);
            this->declare_parameter<int>("gaussian_iterations", 10);
            this->declare_parameter<double>("eta_pose", 0.01);
            this->declare_parameter<double>("eta_gaussian", 0.002);

            gauss_init_size_px_ = this->get_parameter("gauss_init_size_px").as_int();
            gauss_init_scale_ = this->get_parameter("gauss_init_scale").as_double();
            depth_scale_ = this->get_parameter("depth_scale").as_double();

            gs_core_.setGaussInitSizePx(gauss_init_size_px_);
            gs_core_.setGaussInitScale(static_cast<float>(gauss_init_scale_));
            
            // Configurar parámetros de optimización
            gs_core_.setPoseIterations(this->get_parameter("pose_iterations").as_int());
            gs_core_.setGaussianIterations(this->get_parameter("gaussian_iterations").as_int());
            gs_core_.setEtaPose(static_cast<float>(this->get_parameter("eta_pose").as_double()));
            gs_core_.setEtaGaussian(static_cast<float>(this->get_parameter("eta_gaussian").as_double()));

            // Subscribimos los nodos a los topicos
            imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
                imu_preint_topic, rclcpp::SensorDataQoS(),
                std::bind(&GSSlamNode::imuCallback, this, std::placeholders::_1));

            auto sensor_qos = rclcpp::SensorDataQoS();
            depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
                this, depth_topic, sensor_qos.get_rmw_qos_profile());
            color_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
                this, color_topic, sensor_qos.get_rmw_qos_profile());

            // Para el algoritmo es necesario que las imagenes de color y profundidad
            // esten sincronizadas
            rgbd_sync_ = std::make_shared<message_filters::Synchronizer<RGBDSyncPolicy>>(
                RGBDSyncPolicy(10), *color_sub_, *depth_sub_);
            rgbd_sync_->registerCallback(
                std::bind(&GSSlamNode::rgbdCallback, this, std::placeholders::_1, std::placeholders::_2));

            camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
                camera_info_topic, rclcpp::QoS(10).transient_local(),
                std::bind(&GSSlamNode::cameraInfoCallback, this, std::placeholders::_1));

            // Crear publishers de odometría
            odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
            odom_imu_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom_imu", 10);
            
            // Configurar frame_id de mensajes de odometría
            odom_msg_.header.frame_id = world_frame_id_;
            odom_imu_msg_.header.frame_id = world_frame_id_;

            RCLCPP_INFO(this->get_logger(), "Odometry publishers created on topics 'odom' and 'odom_imu'");


            
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
        RCLCPP_DEBUG(this->get_logger(), "CameraInfo received: frame_id=%s",
                     msg->header.frame_id.c_str());
        // Los parametros intrinsecos se representan en K con K =:
        // [fx  0 cx]
        // [ 0 fy cy]
        // [ 0  0  1]
        intrinsics.f = float2{msg->K[0], msg->K[4]};
        intrinsics.c = float2{msg->K[2], msg->K[5]};
        camera_frame_id_ = msg->header.frame_id;

        gs_core_.setIntrinsics(intrinsics);
        hasIntrinsics = true;

        if (!imu_frame_id_.empty() && !camera_frame_id_.empty())
        {
            try
            {
                auto t = tf_buffer_->lookupTransform(
                    imu_frame_id_, camera_frame_id_,
                    tf2::TimePointZero);

                tf2::fromMsg(t.transform.rotation, q_imu_cam_);    // rotate from camera to IMU
                tf2::fromMsg(t.transform.translation, t_imu_cam_); // pos camera in imu frame
                gs_core_.setImuToCamExtrinsics(t_imu_cam_, q_imu_cam_);
                hasCameraInfo = true;
            }
            catch (const tf2::TransformException &ex)
            {
                RCLCPP_WARN(this->get_logger(), "No IMU->camera transform yet: %s", ex.what());
            }
        }
    }

    void GSSlamNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        last_imu_callback_time_ = this->now();
        imu_frame_id_ = msg->header.frame_id;

        RCLCPP_DEBUG(this->get_logger(), "IMU received: frame_id=%s ts=%u.%u",
            msg->header.frame_id.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);

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
                auto t = tf_buffer_->lookupTransform(
                    imu_frame_id_, camera_frame_id_,
                    tf2::TimePointZero);

                tf2::fromMsg(t.transform.rotation, q_imu_cam_);
                tf2::fromMsg(t.transform.translation, t_imu_cam_);
                gs_core_.setImuToCamExtrinsics(t_imu_cam_, q_imu_cam_);
                hasCameraInfo = true;
            }
            catch (const tf2::TransformException &ex)
            {
                RCLCPP_WARN(this->get_logger(), "No IMU->camera transform yet: %s", ex.what());
            }
        }

        // Sin cámara no seguimos para mantener sincronía con RGBD
        if (!hasCameraInfo) {
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
        imu_data_.Acc = raw_acc;
        imu_data_.Gyro = raw_gyro;
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
            odom_pose_init_.position.x = static_cast<float>(t_imu_cam_.x());
            odom_pose_init_.position.y = static_cast<float>(t_imu_cam_.y());
            odom_pose_init_.position.z = static_cast<float>(t_imu_cam_.z());
            odom_pose_init_.orientation.x = static_cast<float>(q_init_cam.w());
            odom_pose_init_.orientation.y = static_cast<float>(q_init_cam.x());
            odom_pose_init_.orientation.z = static_cast<float>(q_init_cam.y());
            odom_pose_init_.orientation.w = static_cast<float>(q_init_cam.z());

            // Configurar imu_data_ completo con medición inicial
            imu_data_.Acc = avg_acc_;
            imu_data_.Gyro = avg_gyro_;
            imu_data_.g_norm = 9.81;
            
            // Inicializar preintegración en GSSlam y en el nodo
            gs_core_.initializeImu(imu_data_);
            preint_.init(imu_data_.Acc, imu_data_.Gyro, acc_bias_, avg_gyro_,
                         imu_data_.acc_n, imu_data_.gyr_n, imu_data_.acc_w, imu_data_.gyr_w);
            
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
            preint_.add_imu(dt, imu_data_.Acc, imu_data_.Gyro);
        }

        const double *pose = gs_core_.getImuPose();
        const double *velocity = gs_core_.getImuVelocity();
        if (pose && velocity)
        {
            Eigen::Map<const Eigen::Vector3d> P(pose);
            Eigen::Quaterniond Q(pose[6], pose[3], pose[4], pose[5]);
            Eigen::Map<const Eigen::Vector3d> V(velocity);

            Eigen::Vector3d pos_imu, vel_imu;
            Eigen::Quaterniond rot_imu;
            preint_.predict(P, Q, V, pos_imu, rot_imu, vel_imu);

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

        syncRgbdImu();
    }

    void GSSlamNode::imuPreintegratedCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        RCLCPP_DEBUG(this->get_logger(), "IMU preintegrated received: frame_id=%s ts=%u.%u",
                     msg->header.frame_id.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);
    }

    void GSSlamNode::rgbdCallback(const sensor_msgs::msg::Image::SharedPtr color,
                                  const sensor_msgs::msg::Image::SharedPtr depth)
    {
        if (!hasCameraInfo) return;

        // Guardamos tiempos de llegada
        last_rgbd_callback_time_ = this->now();
        last_rgb_stamp_ = rclcpp::Time(color->header.stamp);
        last_rgb_msg_ = color;

        RCLCPP_DEBUG(this->get_logger(), "RGBD received: color %ux%u enc=%s | depth %ux%u enc=%s",
            color->width, color->height, color->encoding.c_str(),
            depth->width, depth->height, depth->encoding.c_str());

        
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
        if (isProcessing) return;
        if (!imuInitialized || !hasCameraInfo) return;
        if (last_rgb_stamp_.nanoseconds() == 0) return;

        if (last_rgb_stamp_ <= last_processed_rgbd_stamp_) return;

        rclcpp::Time last_imu_time(imu_cache_preint_.getLatestTime(), RCL_ROS_TIME);
        if (last_imu_time.nanoseconds() == 0) return;
        if (last_imu_time < last_rgb_stamp_) return;

        auto imu_interval = imu_cache_preint_.getInterval(last_processed_rgbd_stamp_, last_rgb_stamp_);
        if (imu_interval.empty()) return;

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
                gs_core_.addImuMeasurement(dt, acc, gyr);
            }
            t_prev = t_cur;
        }

        last_processed_rgbd_stamp_ = last_rgb_stamp_;

        isProcessing = true;

        try
        {
            gs_core_.compute(local_rgb_img, local_depth_img, odom_pose_init_);
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in compute(): %s", e.what());
            isProcessing = false;
            return;
        }

        const double *pose = gs_core_.getImuPose();
        const double *velocity = gs_core_.getImuVelocity();

        if (pose && velocity)
        {
            odom_msg_.header.stamp = last_rgb_stamp_;
            odom_msg_.child_frame_id = last_rgb_msg_ ? last_rgb_msg_->header.frame_id : "camera_link";

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

            odom_pub_->publish(odom_msg_);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "getImuPose() returned null pointer!");
        }

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
            preint_.init(acc, gyr, ba, bg,
                         imu_data_.acc_n, imu_data_.gyr_n, imu_data_.acc_w, imu_data_.gyr_w);

            for (const auto &m : remaining_imu)
            {
                Eigen::Vector3d acc_i(m->linear_acceleration.x,
                                      m->linear_acceleration.y,
                                      m->linear_acceleration.z);
                Eigen::Vector3d gyr_i(m->angular_velocity.x,
                                      m->angular_velocity.y,
                                      m->angular_velocity.z);
                preint_.add_imu((rclcpp::Time(m->header.stamp, RCL_ROS_TIME) - t_imu).seconds(),
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
            gs_core_.compute(local_rgb_img, local_depth_img, CameraPose());
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error in compute(): %s", e.what());
            isProcessing = false;
            return;
        }

        // TODO: integrar imu_buffer_ con las imágenes sincronizadas
        // - Procesar preintegración usando dt de cada muestra IMU
        // - Fusionar con pose visual del gs_core_

        // Publicar odometría después de procesar frame
        const double *pose = gs_core_.getImuPose();
        const double *velocity = gs_core_.getImuVelocity();
        
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