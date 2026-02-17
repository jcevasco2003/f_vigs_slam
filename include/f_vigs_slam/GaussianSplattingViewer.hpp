#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <atomic>
#include <thread>
#include <filesystem>
#include <iostream>

#include "f_vigs_slam/GSSlam.cuh"

namespace f_vigs_slam
{
    class GaussianSplattingViewer
    {
    public:
        explicit GaussianSplattingViewer(GSSlam &gs_slam);
        ~GaussianSplattingViewer();

        void startThread();
        void stop();

    protected:
        static void mouseCallbackStatic(int event, int x, int y, int flags, void *userdata);
        void mouseCallback(int event, int x, int y, int flags);
        void keyCallback(int key);
        void renderLoop();
        void render();
        void resetView();
        void saveFrame(const cv::Mat &frame, const std::string &type);

        GSSlam &gs_slam_;

        cv::Mat rendered_rgb_;
        cv::Mat rendered_depth_;
        cv::cuda::GpuMat rendered_rgb_gpu_;
        cv::cuda::GpuMat rendered_depth_gpu_;

        CameraPose camera_pose_;
        IntrinsicParameters camera_intrinsics_;
        int width_;
        int height_;
        double fov_;

        bool follow_ = true;

        Eigen::Vector3f camera_view_position_;
        double yaw_;
        double pitch_;
        Eigen::Vector3f focal_point_;
        double distance_;

        int prev_mouse_x_;
        int prev_mouse_y_;

        std::thread render_thread_;
        std::atomic<bool> stop_{false};
        bool gui_enabled_ = true;

        // Frame saving
        int frame_counter_ = 0;
        std::string frames_dir_ = "frames";

        enum {
            RENDER_TYPE_RGB = 0,
            RENDER_TYPE_DEPTH = 1,
            RENDER_TYPE_BLOBS = 2,
            RENDER_TYPE_NUM
        } render_type_ = RENDER_TYPE_RGB;
    };
} // namespace f_vigs_slam
