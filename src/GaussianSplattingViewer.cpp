#include <f_vigs_slam/GaussianSplattingViewer.hpp>
#include <cmath>
#include <iomanip>
#include <chrono>

namespace f_vigs_slam
{
    GaussianSplattingViewer::GaussianSplattingViewer(GSSlam &gs_slam)
        : gs_slam_(gs_slam),
          width_(848),
          height_(480),
          fov_(60.0),
          yaw_(0.0),
          pitch_(0.2),
          focal_point_(2.0f, 0.0f, 0.0f),
          distance_(3.0),
          prev_mouse_x_(0),
          prev_mouse_y_(0)
    {
        resetView();
        // Create frames directory if it doesn't exist
        try {
            std::filesystem::create_directories(frames_dir_);
            std::cout << "[GaussianSplattingViewer] Frames will be saved to: " << frames_dir_ << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[GaussianSplattingViewer] Error creating frames directory: " << e.what() << std::endl;
        }
    }

    GaussianSplattingViewer::~GaussianSplattingViewer()
    {
        stop();
    }

    void GaussianSplattingViewer::resetView()
    {
        focal_point_ = Eigen::Vector3f(2.0f, 0.0f, 0.0f);
        distance_ = 3.0;
        camera_view_position_ = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        yaw_ = 0.0;
        pitch_ = 0.2;
        fov_ = 60.0;
    }

    void GaussianSplattingViewer::startThread()
    {
        if (render_thread_.joinable()) {
            return;
        }
        stop_.store(false);
        render_thread_ = std::thread(&GaussianSplattingViewer::renderLoop, this);
    }

    void GaussianSplattingViewer::stop()
    {
        stop_.store(true);
        if (render_thread_.joinable()) {
            render_thread_.join();
        }
    }

    void GaussianSplattingViewer::mouseCallbackStatic(int event, int x, int y, int flags, void *userdata)
    {
        auto *viewer = static_cast<GaussianSplattingViewer *>(userdata);
        viewer->mouseCallback(event, x, y, flags);
    }

    void GaussianSplattingViewer::mouseCallback(int event, int x, int y, int flags)
    {
        if (event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_RBUTTONDOWN || event == cv::EVENT_MBUTTONDOWN)
        {
            prev_mouse_x_ = x;
            prev_mouse_y_ = y;
        }
        else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON))
        {
            int dx = x - prev_mouse_x_;
            int dy = y - prev_mouse_y_;

            yaw_ -= dx * 0.01;
            pitch_ += dy * 0.01;

            prev_mouse_x_ = x;
            prev_mouse_y_ = y;
        }
        else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_RBUTTON))
        {
            int dx = x - prev_mouse_x_;
            int dy = y - prev_mouse_y_;

            Eigen::Quaternionf orientation =
                Eigen::AngleAxisf(static_cast<float>(yaw_), Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(static_cast<float>(pitch_), Eigen::Vector3f::UnitY());
            camera_view_position_ += orientation * Eigen::Vector3f(0.0f, dx * 0.01f, dy * 0.01f);

            prev_mouse_x_ = x;
            prev_mouse_y_ = y;
        }
        else if (event == cv::EVENT_MOUSEWHEEL)
        {
            int delta = cv::getMouseWheelDelta(flags);
            Eigen::Quaternionf orientation =
                Eigen::AngleAxisf(static_cast<float>(yaw_), Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(static_cast<float>(pitch_), Eigen::Vector3f::UnitY());
            camera_view_position_ += orientation * Eigen::Vector3f(-delta * 0.1f, 0.0f, 0.0f);
        }
    }

    void GaussianSplattingViewer::keyCallback(int key)
    {
        if (key == int(' '))
        {
            render_type_ = static_cast<decltype(render_type_)>((render_type_ + 1) % RENDER_TYPE_NUM);
        }
        else if (key == int('r'))
        {
            resetView();
        }
        else if (key == int('f'))
        {
            resetView();
            follow_ = !follow_;
        }
        else if (key == 82 || key == 84)
        {
            float d = (key == 82 ? 0.1f : -0.1f);
            Eigen::Quaternionf orientation =
                Eigen::AngleAxisf(static_cast<float>(yaw_), Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(static_cast<float>(pitch_), Eigen::Vector3f::UnitY());
            camera_view_position_ += orientation * Eigen::Vector3f(d, 0.0f, 0.0f);
        }
        else if (key == 81 || key == 83)
        {
            float d = (key == 81 ? 0.1f : -0.1f);
            Eigen::Quaternionf orientation =
                Eigen::AngleAxisf(static_cast<float>(yaw_), Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(static_cast<float>(pitch_), Eigen::Vector3f::UnitY());
            camera_view_position_ += orientation * Eigen::Vector3f(0.0f, d, 0.0f);
        }
    }

    void GaussianSplattingViewer::renderLoop()
    {
        try {
            cv::namedWindow("Gaussian Splatting Viewer", cv::WINDOW_AUTOSIZE);
            cv::setMouseCallback("Gaussian Splatting Viewer", mouseCallbackStatic, this);
            gui_enabled_ = true;
        } catch (const cv::Exception& e) {
            gui_enabled_ = false;
            std::cerr << "Warning: OpenCV GUI not available (no GTK support). Viewer disabled." << std::endl;
            std::cerr << "  Error: " << e.what() << std::endl;
            std::cerr << "  To enable viewer, recompile OpenCV with: -D WITH_GTK=ON" << std::endl;
        }

        while (!stop_.load())
        {
            render();
            if (gui_enabled_) {
                int key = cv::waitKey(30);
                if (key >= 0) {
                    keyCallback(key);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        }

        if (gui_enabled_) {
            cv::destroyWindow("Gaussian Splatting Viewer");
        }
    }

    void GaussianSplattingViewer::render()
    {
        int img_w = gs_slam_.getImageWidth();
        int img_h = gs_slam_.getImageHeight();
        if (img_w > 0 && img_h > 0) {
            width_ = img_w;
            height_ = img_h;
        }

        if (width_ <= 0 || height_ <= 0) {
            return;
        }

        Eigen::Quaternionf pitchyaw =
            Eigen::AngleAxisf(static_cast<float>(yaw_), Eigen::Vector3f::UnitZ()) *
            Eigen::AngleAxisf(static_cast<float>(pitch_), Eigen::Vector3f::UnitY());
        Eigen::Quaternionf camera_orientation = pitchyaw * Eigen::Quaternionf(-0.5f, 0.5f, -0.5f, 0.5f);

        camera_pose_.orientation = make_float4(camera_orientation.w(),
                                               camera_orientation.x(),
                                               camera_orientation.y(),
                                               camera_orientation.z());
        camera_pose_.position = make_float3(camera_view_position_.x(),
                                            camera_view_position_.y(),
                                            camera_view_position_.z());

        float fx = static_cast<float>(width_ / (2.0 * std::tan(fov_ * M_PI / 360.0)));
        camera_intrinsics_.f = make_float2(fx, fx);
        camera_intrinsics_.c = make_float2(static_cast<float>(width_) * 0.5f,
                                           static_cast<float>(height_) * 0.5f);

        if (!gs_slam_.renderView(camera_pose_, camera_intrinsics_, width_, height_,
                                 rendered_rgb_gpu_, rendered_depth_gpu_)) {
            return;
        }

        if (render_type_ == RENDER_TYPE_RGB)
        {
            rendered_rgb_gpu_.download(rendered_rgb_);
            if (gui_enabled_) {
                cv::imshow("Gaussian Splatting Viewer", rendered_rgb_);
            }
            // Save frame every 10 renders (adjust as needed)
            if (frame_counter_ % 10 == 0) {
                saveFrame(rendered_rgb_, "rgb");
            }
        }
        else if (render_type_ == RENDER_TYPE_DEPTH)
        {
            rendered_depth_gpu_.download(rendered_depth_);
            if (gui_enabled_) {
                cv::imshow("Gaussian Splatting Viewer", 0.15 * rendered_depth_);
            }
            if (frame_counter_ % 10 == 0) {
                saveFrame(0.15 * rendered_depth_, "depth");
            }
        }
        else if (render_type_ == RENDER_TYPE_BLOBS)
        {
            rendered_rgb_gpu_.download(rendered_rgb_);
            if (gui_enabled_) {
                cv::imshow("Gaussian Splatting Viewer", rendered_rgb_);
            }
            if (frame_counter_ % 10 == 0) {
                saveFrame(rendered_rgb_, "blobs");
            }
        }
        frame_counter_++;
    }

    void GaussianSplattingViewer::saveFrame(const cv::Mat &frame, const std::string &type)
    {
        if (frame.empty()) {
            return;
        }

        try {
            // Create subdirectory for this type
            std::string subdir = frames_dir_ + "/" + type;
            std::filesystem::create_directories(subdir);

            // Generate filename with index
            std::ostringstream oss;
            oss << subdir << "/frame_" << std::setfill('0') << std::setw(6) << frame_counter_ << ".png";
            std::string filename = oss.str();

            // Save the frame
            if (cv::imwrite(filename, frame)) {
                std::cout << "[GaussianSplattingViewer] Saved: " << filename << std::endl;
            } else {
                std::cerr << "[GaussianSplattingViewer] Failed to save: " << filename << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[GaussianSplattingViewer] Error saving frame: " << e.what() << std::endl;
        }
    }
} // namespace f_vigs_slam
