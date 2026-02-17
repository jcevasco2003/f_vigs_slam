#include <f_vigs_slam/GSSlam.cuh>
#include <f_vigs_slam/GSCudaKernels.cuh>
#include <f_vigs_slam/RgbdPoseCost.hpp>
#include <f_vigs_slam/ImuCostFunction.hpp>
#include <f_vigs_slam/PoseLocalParameterization.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <ceres/ceres.h>
#include <chrono>
#include <random>
#include <thread>

// Implementamos el codigo principal en CUDA para todas las funciones paralelizables

namespace f_vigs_slam
{
    namespace
    {
        struct ToFloat3
        {
            __host__ __device__ float3 operator()(const thrust::tuple<float2, float> &t) const
            {
                float2 p = thrust::get<0>(t);
                float z = thrust::get<1>(t);
                return make_float3(p.x, p.y, z);
            }
        };

        Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v)
        {
            Eigen::Matrix3d m;
            m << 0.0, -v.z(), v.y(),
                 v.z(), 0.0, -v.x(),
                -v.y(), v.x(), 0.0;
            return m;
        }
    }

    GSSlam::GSSlam()
        :   n_Gaussians(0),
            max_Gaussians(10000),
            isFirstImage(true),
            isInitialized(false),
            first_image_(true),
            nb_images_processed_(0)
    {
        gaussians_.positions.resize(max_Gaussians);
        gaussians_.colors.resize(max_Gaussians);
        gaussians_.orientations.resize(max_Gaussians);
        gaussians_.scales.resize(max_Gaussians);
        gaussians_.opacities.resize(max_Gaussians);

        instance_counter_.resize(1);
        cudaMemset(thrust::raw_pointer_cast(instance_counter_.data()), 0, sizeof(uint32_t));

        // Inicializar gradientes
        gaussian_gradients_.resize(max_Gaussians);
        opacity_gradients_.resize(max_Gaussians);
        
        // Configurar tiles (16x16 pixels por tile)
        tile_size_ = make_uint2(16, 16);
        bg_color_ = make_float3(0.0f, 0.0f, 0.0f);  // Negro
        
        // Inicializar contador de instancias en screen-space
        instance_counter_screen_.resize(1);

        // Inicializar estado IMU: pose en origen, orientación identidad
        P_cur_[0] = P_cur_[1] = P_cur_[2] = 0.0;        // posición
        P_cur_[3] = P_cur_[4] = P_cur_[5] = 0.0;        // qx, qy, qz
        P_cur_[6] = 1.0;                                 // qw
        
        P_prev_[0] = P_prev_[1] = P_prev_[2] = 0.0;
        P_prev_[3] = P_prev_[4] = P_prev_[5] = 0.0;
        P_prev_[6] = 1.0;
        
        // Inicializar velocidad y biases a cero
        for (int i = 0; i < 9; ++i) {
            VB_cur_[i] = 0.0;
            VB_prev_[i] = 0.0;
        }
        
        // Inicializar pirámides
        pyr_color_.resize(nb_pyr_levels_);
        pyr_depth_.resize(nb_pyr_levels_);
        pyr_dx_.resize(nb_pyr_levels_);
        pyr_dy_.resize(nb_pyr_levels_);
        
        // Inicializar preintegración IMU
        preint_ = new Preintegration();
        preint_shared_ = std::shared_ptr<Preintegration>(preint_, [](Preintegration*) {});

        // Construir problema Ceres una sola vez (reutilizable)
        visual_cost_ = new RgbdPoseCostFunction(this);
        imu_cost_ = new ImuCostFunction(preint_shared_);
        marginalization_cost_ = new MarginalizationFactor();

        problem_.AddParameterBlock(P_prev_, 7, new PoseLocalParameterization());
        problem_.AddParameterBlock(P_cur_, 7);
        problem_.SetParameterization(P_cur_, new PoseLocalParameterization());
        problem_.AddParameterBlock(VB_prev_, 9);
        problem_.AddParameterBlock(VB_cur_, 9);

        problem_.AddResidualBlock(visual_cost_, nullptr, P_cur_);
        problem_.AddResidualBlock(imu_cost_, nullptr, P_prev_, VB_prev_, P_cur_, VB_cur_);
        imu_residual_added_ = true;

        marginalization_info_.addResidualBlockInfo(
            new ResidualBlockInfo(visual_cost_, nullptr, {P_cur_}, {}));
        marginalization_info_.addResidualBlockInfo(
            new ResidualBlockInfo(imu_cost_, nullptr, {P_prev_, VB_prev_, P_cur_, VB_cur_}, {0, 1}));

        marginalization_info_.init();

        std::unordered_map<long, double*> addr_shift;
        addr_shift[reinterpret_cast<long>(P_cur_)] = P_prev_;
        addr_shift[reinterpret_cast<long>(VB_cur_)] = VB_prev_;
        std::vector<double*> params = marginalization_info_.getParameterBlocks(addr_shift);

        marginalization_cost_->init(&marginalization_info_);
        marginalization_info_.addResidualBlockInfo(
            new ResidualBlockInfo(marginalization_cost_, nullptr, params, {}));
        problem_.AddResidualBlock(marginalization_cost_, nullptr, params);

        options_.linear_solver_type = ceres::DENSE_QR;
        options_.minimizer_progress_to_stdout = false;
        // Relajamos tolerancia
        options_.function_tolerance = 1e-5;    // por defecto 1e-6
        options_.gradient_tolerance = 1e-9;    // por defecto 1e-10
        options_.parameter_tolerance = 1e-7;   // por defecto 1e-8

        isInitialized = true;
    }

    GSSlam::~GSSlam()
    {
        stop_optimization_.store(true);
        if (optimize_thread_.joinable()) {
            optimize_thread_.join();
        }
        if (preint_) {
            delete preint_;
            preint_ = nullptr;
        }
    }

    void GSSlam::setIntrinsics(const IntrinsicParameters &params)
    {
        intrinsics_ = params;
        intrinsics_set_ = true;
    }

    void GSSlam::setGaussInitSizePx(int size_px)
    {
        gauss_init_size_px_ = std::max(1, size_px);
    }

    void GSSlam::setGaussInitScale(float scale)
    {
        gauss_init_scale_ = std::max(1e-5f, scale);
    }

    void GSSlam::setImuToCamExtrinsics(const Eigen::Vector3d &t_imu_cam,
                                       const Eigen::Quaterniond &q_imu_cam)
    {
        t_imu_cam_ = t_imu_cam;
        q_imu_cam_ = q_imu_cam.normalized();
    }

    bool GSSlam::hasGaussians() const
    {
        return n_Gaussians > 0;
    }

    uint32_t GSSlam::getGaussiansCount() const
    {
        return n_Gaussians;
    }

    void GSSlam::ensureCapacity(uint32_t required)
    {
        if (required <= max_Gaussians) return;
        max_Gaussians = required;
        gaussians_.positions.resize(max_Gaussians);
        gaussians_.colors.resize(max_Gaussians);
        gaussians_.orientations.resize(max_Gaussians);
        gaussians_.scales.resize(max_Gaussians);
        gaussians_.opacities.resize(max_Gaussians);
    }

    void GSSlam::initializeGaussiansFromRgbd(const cv::Mat &rgb,
                                             const cv::Mat &depth,
                                             const CameraPose &cameraPose,
                                             float depth_scale)
    {
        if (!intrinsics_set_) return;
        if (rgb.empty() || depth.empty()) return;

        cv::Mat rgb_bgr;
        if (rgb.type() == CV_8UC3)
        {
            rgb_bgr = rgb;
        }
        else
        {
            rgb.convertTo(rgb_bgr, CV_8UC3);
        }

        cv::Mat depth_float;
        if (depth.type() == CV_32FC1)
        {
            depth_float = depth;
        }
        else if (depth.type() == CV_16UC1)
        {
            depth.convertTo(depth_float, CV_32FC1, depth_scale);
        }
        else
        {
            return;
        }

        cv::cuda::GpuMat rgb_gpu, depth_gpu;
        rgb_gpu.upload(rgb_bgr);
        depth_gpu.upload(depth_float);

        cudaMemset(thrust::raw_pointer_cast(instance_counter_.data()), 0, sizeof(uint32_t));

        int width = rgb_gpu.cols;
        int height = rgb_gpu.rows;
        int sample_w = (width + gauss_init_size_px_ - 1) / gauss_init_size_px_;
        int sample_h = (height + gauss_init_size_px_ - 1) / gauss_init_size_px_;

        dim3 block(16, 16);
        dim3 grid((sample_w + block.x - 1) / block.x,
                  (sample_h + block.y - 1) / block.y);

        initGaussiansFromRgbd_kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(gaussians_.positions.data()),
            thrust::raw_pointer_cast(gaussians_.scales.data()),
            thrust::raw_pointer_cast(gaussians_.orientations.data()),
            thrust::raw_pointer_cast(gaussians_.colors.data()),
            thrust::raw_pointer_cast(gaussians_.opacities.data()),
            thrust::raw_pointer_cast(instance_counter_.data()),
            max_Gaussians,
            rgb_gpu.ptr<uchar3>(),
            rgb_gpu.step,
            depth_gpu.ptr<float>(),
            depth_gpu.step,
            width,
            height,
            intrinsics_,
            cameraPose,
            static_cast<uint32_t>(gauss_init_size_px_),
            static_cast<uint32_t>(gauss_init_size_px_),
            gauss_init_scale_,
            gauss_init_opacity_);

        cudaDeviceSynchronize();

        uint32_t host_count = 0;
        cudaMemcpy(&host_count,
                   thrust::raw_pointer_cast(instance_counter_.data()),
                   sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);

        n_Gaussians = std::min(host_count, max_Gaussians);
        
        // Actualizar pose actual y estado IMU
        current_pose_ = cameraPose;
        
        // Actualizar P_cur_ con la pose de la cámara
        P_cur_[0] = cameraPose.position.x;
        P_cur_[1] = cameraPose.position.y;
        P_cur_[2] = cameraPose.position.z;
        P_cur_[3] = cameraPose.orientation.y;  // qx
        P_cur_[4] = cameraPose.orientation.z;  // qy
        P_cur_[5] = cameraPose.orientation.w;  // qz
        P_cur_[6] = cameraPose.orientation.x;  // qw
    }

    // ===== CORE SLAM METHODS =====
    
    void GSSlam::prepareRasterization(const CameraPose &camera_pose, 
                                       const IntrinsicParameters &intrinsics,
                                       int width, int height)
    {
        if (n_Gaussians == 0) return;
        
        // ============================================================
        // PASO 1: Calcular número de tiles
        // ============================================================
        num_tiles_ = make_uint2(
            (width + tile_size_.x - 1) / tile_size_.x,
            (height + tile_size_.y - 1) / tile_size_.y
        );
        uint32_t num_tiles_total = num_tiles_.x * num_tiles_.y;
        
        // ============================================================
        // PASO 2: Redimensionar buffers
        // ============================================================
        positions_2d_.resize(n_Gaussians);
        covariances_2d_.resize(n_Gaussians);
        inv_covariances_2d_.resize(n_Gaussians);
        depths_.resize(n_Gaussians);
        p_hats_.resize(n_Gaussians);
        
        uint32_t max_instances = n_Gaussians * 9;
        tile_counts_.resize(n_Gaussians);
        tile_offsets_.resize(n_Gaussians + 1);
        hashes_.resize(max_instances);
        gaussian_indices_.resize(max_instances);
        
        tile_ranges_.resize(num_tiles_total);
        cudaMemset(thrust::raw_pointer_cast(tile_ranges_.data()), 0, 
                   num_tiles_total * sizeof(uint2));
        
        // ============================================================
        // PASO 3: Proyectar gaussianas a screen-space 2D
        // ============================================================
        dim3 block(256);
        dim3 grid((n_Gaussians + block.x - 1) / block.x);
        
        projectGaussiansWorldToScreen_kernel<<<grid, block>>>
        (thrust::raw_pointer_cast(positions_2d_.data()),
         thrust::raw_pointer_cast(covariances_2d_.data()),
         thrust::raw_pointer_cast(inv_covariances_2d_.data()),
         thrust::raw_pointer_cast(depths_.data()),
         thrust::raw_pointer_cast(p_hats_.data()),
         thrust::raw_pointer_cast(gaussians_.positions.data()),
         thrust::raw_pointer_cast(gaussians_.scales.data()),
         thrust::raw_pointer_cast(gaussians_.orientations.data()),
         camera_pose,
         intrinsics,
         width,
         height,
         n_Gaussians);
        cudaDeviceSynchronize();
        
        // ============================================================
        // PASO 4: Contar tiles por gaussiana
        // ============================================================
        countTilesPerGaussian_kernel<<<grid, block>>>
        (thrust::raw_pointer_cast(tile_counts_.data()),
         thrust::raw_pointer_cast(positions_2d_.data()),
         thrust::raw_pointer_cast(covariances_2d_.data()),
         width,
         height,
         n_Gaussians,
         num_tiles_.x,
         num_tiles_.y,
         3.0f);
        cudaDeviceSynchronize();
        
        // ============================================================
        // PASO 5: Exclusive scan para obtener offsets
        // ============================================================
        if (tile_counts_.empty() || tile_offsets_.empty() || tile_counts_.size() != static_cast<size_t>(n_Gaussians)) {
            return;
        }
        
        thrust::exclusive_scan(
            tile_counts_.begin(),
            tile_counts_.end(),
            tile_offsets_.begin(),
            0u
        );
        
        if (n_Gaussians <= 0 || static_cast<size_t>(n_Gaussians - 1) >= tile_offsets_.size() || 
            static_cast<size_t>(n_Gaussians - 1) >= tile_counts_.size()) {
            return;
        }
        
        uint32_t nb_instances = tile_offsets_[n_Gaussians - 1] + tile_counts_[n_Gaussians - 1];
        if (nb_instances == 0) return;
        
        if (hashes_.size() < nb_instances) {
            hashes_.resize(nb_instances);
            gaussian_indices_.resize(nb_instances);
        }
        
        // ============================================================
        // PASO 6: Generar hashes para cada tile cubierto
        // ============================================================
        generateTileHashes_kernel<<<grid, block>>>
        (thrust::raw_pointer_cast(hashes_.data()),
         thrust::raw_pointer_cast(gaussian_indices_.data()),
         thrust::raw_pointer_cast(tile_offsets_.data()),
         thrust::raw_pointer_cast(positions_2d_.data()),
         thrust::raw_pointer_cast(covariances_2d_.data()),
         thrust::raw_pointer_cast(depths_.data()),
         thrust::raw_pointer_cast(p_hats_.data()),
         width,
         height,
         n_Gaussians,
         num_tiles_.x,
         num_tiles_.y,
         3.0f);
        cudaDeviceSynchronize();
        
        // ============================================================
        // PASO 7: Ordenar por hash
        // ============================================================
        // CRITICAL VALIDATION: Verificar que los iteradores son válidos
        if (nb_instances > hashes_.size() || nb_instances > gaussian_indices_.size()) {
            std::cerr << "ERROR: sort_by_key - Invalid size mismatch!" << std::endl;
            std::cerr << "  nb_instances: " << nb_instances << std::endl;
            std::cerr << "  hashes_.size(): " << hashes_.size() << std::endl;
            std::cerr << "  gaussian_indices_.size(): " << gaussian_indices_.size() << std::endl;
            return;
        }
        
        if (nb_instances == 0) {
            return;
        }
        
        thrust::sort_by_key(
            hashes_.begin(),
            hashes_.begin() + nb_instances,
            gaussian_indices_.begin()
        );
        
        // ============================================================
        // PASO 8: Calcular rangos de gaussianas por tile
        // ============================================================
        dim3 tile_grid((num_tiles_total + block.x - 1) / block.x);
        computeIndicesRanges_kernel<<<tile_grid, block>>>
        (thrust::raw_pointer_cast(tile_ranges_.data()),
         thrust::raw_pointer_cast(hashes_.data()),
         nb_instances,
         num_tiles_total);
        cudaDeviceSynchronize();
        
        last_nb_instances_ = nb_instances;
    }
    
    void GSSlam::rasterize(const CameraPose &camera_pose, 
                           const IntrinsicParameters &intrinsics,
                           int width, int height)
    {
        if (n_Gaussians == 0) return;
        
        // ============================================================
        // PASO 1: Preparar screen-space transform y sorting
        // ============================================================
        prepareRasterization(camera_pose, intrinsics, width, height);
        
        if (last_nb_instances_ == 0) return;
        
        // Validar integridad de buffers críticos
        if (gaussian_indices_.empty() || tile_ranges_.empty() || 
            positions_2d_.empty() || covariances_2d_.empty() ||
            depths_.empty() || p_hats_.empty()) {
            std::cerr << "ERROR in rasterize(): Buffers vacios tras prepareRasterization()" << std::endl;
            return;
        }
        
        // ============================================================
        // PASO 2: Asegurar que buffers de salida existen
        // ============================================================
        if (rendered_rgb_gpu_.empty() || 
            rendered_rgb_gpu_.cols != width || 
            rendered_rgb_gpu_.rows != height)
        {
            rendered_rgb_gpu_.create(height, width, CV_8UC3);
            rendered_depth_gpu_.create(height, width, CV_32FC1);
        }
        
        // Limpiar buffers (negro + depth infinito)
        rendered_rgb_gpu_.setTo(cv::Scalar(0, 0, 0));
        rendered_depth_gpu_.setTo(cv::Scalar(1e10f));
        
        // ============================================================
        // PASO 3: Rasterizar gaussianas (tile-based rendering)
        // ============================================================
        dim3 block(tile_size_.x, tile_size_.y);  // 16x16
        dim3 grid(num_tiles_.x, num_tiles_.y);
        
        if (grid.x == 0 || grid.y == 0) {
            std::cerr << "ERROR in rasterize(): Invalid grid dimensions" << std::endl;
            return;
        }
        
        forwardPassTileKernel<<<grid, block>>>
        ((float3*)rendered_rgb_gpu_.ptr<uchar3>(),
         rendered_depth_gpu_.ptr<float>(),
         thrust::raw_pointer_cast(gaussian_indices_.data()),
         thrust::raw_pointer_cast(tile_ranges_.data()),
         thrust::raw_pointer_cast(positions_2d_.data()),
         thrust::raw_pointer_cast(covariances_2d_.data()),
         thrust::raw_pointer_cast(gaussians_.colors.data()),
         thrust::raw_pointer_cast(gaussians_.opacities.data()),
         thrust::raw_pointer_cast(depths_.data()),
         thrust::raw_pointer_cast(p_hats_.data()),
         width,
         height,
         num_tiles_.x,
         num_tiles_.y);
        cudaDeviceSynchronize();
    }

    bool GSSlam::renderView(const CameraPose &camera_pose,
                            const IntrinsicParameters &intrinsics,
                            int width, int height,
                            cv::cuda::GpuMat &rendered_rgb,
                            cv::cuda::GpuMat &rendered_depth)
    {
        std::lock_guard<std::mutex> lock(optimization_mutex_);

        if (!intrinsics_set_ || n_Gaussians == 0) {
            return false;
        }
        if (width <= 0 || height <= 0) {
            return false;
        }

        rasterize(camera_pose, intrinsics, width, height);

        if (rendered_rgb_gpu_.empty() || rendered_depth_gpu_.empty()) {
            return false;
        }

        rendered_rgb = rendered_rgb_gpu_;
        rendered_depth = rendered_depth_gpu_;
        return true;
    }

    void GSSlam::updateCameraPoseFromImu()
    {
        Eigen::Vector3f imu_trans(static_cast<float>(P_cur_[0]),
                                  static_cast<float>(P_cur_[1]),
                                  static_cast<float>(P_cur_[2]));
        Eigen::Quaternionf imu_rot(static_cast<float>(P_cur_[6]),
                                   static_cast<float>(P_cur_[3]),
                                   static_cast<float>(P_cur_[4]),
                                   static_cast<float>(P_cur_[5]));
        Eigen::Vector3f cam_trans = imu_trans + imu_rot.normalized().toRotationMatrix() * t_imu_cam_.cast<float>();
        Eigen::Quaternionf cam_rot = imu_rot * q_imu_cam_.cast<float>();
        cam_rot.normalize();

        current_pose_.position = make_float3(cam_trans.x(), cam_trans.y(), cam_trans.z());
        current_pose_.orientation = make_float4(cam_rot.w(), cam_rot.x(), cam_rot.y(), cam_rot.z());
    }
    
    void GSSlam::compute(const cv::Mat &rgb, const cv::Mat &depth, const CameraPose &odometry_pose)
    {
        if (!intrinsics_set_) {
            return;
        }

        std::lock_guard<std::mutex> lock(optimization_mutex_);
        
        // ============================================================
        // PASO 1: Inicializar y copiar imágenes a GPU (con pirámides)
        // ============================================================
        initAndCopyImgs(rgb, depth);
        
        // ============================================================
        // PASO 2: Primera imagen - Inicialización
        // ============================================================
        if (first_image_) {
            // Convertimos odometry_pose (x, y, z, w) a CameraPose interno (w, x, y, z)
            CameraPose odom_pose_converted = odometry_pose;
            odom_pose_converted.orientation = make_float4(
                odometry_pose.orientation.w,
                odometry_pose.orientation.x,
                odometry_pose.orientation.y,
                odometry_pose.orientation.z);

            // Inicializar pose IMU
            P_cur_[0] = odom_pose_converted.position.x;
            P_cur_[1] = odom_pose_converted.position.y;
            P_cur_[2] = odom_pose_converted.position.z;
            // CameraPose interno usa quaternion (w, x, y, z)
            P_cur_[3] = odom_pose_converted.orientation.y;  // qx
            P_cur_[4] = odom_pose_converted.orientation.z;  // qy
            P_cur_[5] = odom_pose_converted.orientation.w;  // qz
            P_cur_[6] = odom_pose_converted.orientation.x;  // qw
            
            // Copiar a P_prev
            for (int i = 0; i < 7; ++i) {
                P_prev_[i] = P_cur_[i];
            }
            
            // Pose de cámara
            current_pose_ = odom_pose_converted;
            
            // Generar gaussianas desde RGB-D
            initializeGaussiansFromRgbd(rgb, depth, odom_pose_converted, 0.001f);
            
            // Agregar primer keyframe
            addKeyframe();

            if (!optimize_thread_.joinable()) {
                stop_optimization_.store(false);
                optimize_thread_ = std::thread(&GSSlam::optimizationLoop, this);
            }
            
            first_image_ = false;
        }
        
        // ============================================================
        // PASO 3: Verificar si IMU está inicializada
        // ============================================================
        if (imu_initialized_ && preint_->is_initialized) {
            // ============================================================
            // PASO 3.1: Predicción IMU usando preintegración
            // ============================================================
            Eigen::Map<Eigen::Quaterniond> Q_cur_eig(P_cur_ + 3);
            Eigen::Map<Eigen::Vector3d> P_cur_eig(P_cur_);
            Eigen::Map<Eigen::Quaterniond> Q_prev_eig(P_prev_ + 3);
            Eigen::Map<Eigen::Vector3d> P_prev_eig(P_prev_);
            Eigen::Map<Eigen::VectorXd> VB_cur_eig(VB_cur_, 9);
            Eigen::Map<Eigen::VectorXd> VB_prev_eig(VB_prev_, 9);
            Eigen::Map<Eigen::Vector3d> V_prev_eig(VB_prev_);
            Eigen::Map<Eigen::Vector3d> V_cur_eig(VB_cur_);
            
            Eigen::Vector3d Pj, Vj;
            Eigen::Quaterniond Qj;
            
            // Predicción IMU
            preint_->predict(P_prev_eig, Q_prev_eig, V_prev_eig, Pj, Qj, Vj);
            
            // Actualizar estado predicho
            P_cur_eig = Pj;
            Q_cur_eig = Qj;
            V_cur_eig = Vj;
        
            // ============================================================
            // PASO 4: Warping inicial (opcional)
            // ============================================================
            // Reproyecta el frame previo al pose predicho para mejorar la
            // convergencia del estimador visual (cuando se usa warping).
            {
                Eigen::Vector3d P_cam = Pj + Qj * t_imu_cam_;
                Eigen::Quaterniond Q_cam = Qj * q_imu_cam_;
                CameraPose predicted_pose;
                predicted_pose.position = make_float3(static_cast<float>(P_cam.x()),
                                                     static_cast<float>(P_cam.y()),
                                                     static_cast<float>(P_cam.z()));
                predicted_pose.orientation = make_float4(static_cast<float>(Q_cam.w()),
                                                         static_cast<float>(Q_cam.x()),
                                                         static_cast<float>(Q_cam.y()),
                                                         static_cast<float>(Q_cam.z()));
                initWarping(predicted_pose);
            }
            
            // ============================================================
            // PASO 5: Optimización de pose con Ceres Solver
            // ============================================================
            // Combina factores visual, IMU y prior (si existe)
            // Tracking: solo niveles coarse (omite nivel 0)
            optimizeWithCeres(1, pose_iterations_);
            
            // ============================================================
            // PASO 6: Actualizar pose de cámara desde IMU
            // ============================================================
            updateCameraPoseFromImu();
            
            // ============================================================
            // PASO 7: Remover outliers
            // ============================================================
            removeOutliers();
            
            // ============================================================
            // PASO 8: Verificar covisibilidad para nuevo keyframe
            // ============================================================
            float covis_ratio = computeCovisibilityRatio();
            
            if (covis_ratio < covisibility_threshold_) {
                // Nuevo keyframe necesario - optimización refinada
                std::cout << "Nuevo keyframe (covisibilidad = " << covis_ratio << ")" << std::endl;
                
                // Optimización completa con Ceres para keyframe (incluye nivel 0)
                optimizeWithCeres(0, std::max(1, pose_iterations_ * 2));
                
                // Actualizar pose de cámara nuevamente
                updateCameraPoseFromImu();
                
                // Gestión de mapa
                prune();
                addKeyframe();
                densify(keyframes_[current_keyframe_idx_]);
            }
            
            // ============================================================
            // PASO 9: Marginalización y actualización de estado
            // ============================================================
            marginalization_info_.preMarginalize();
            marginalization_info_.marginalize();

            // Actualizar estado previo
            Q_prev_eig = Q_cur_eig;
            P_prev_eig = P_cur_eig;
            VB_prev_eig = VB_cur_eig;
            
            // Reinicializar preintegración IMU para próximo frame
            preint_->init(last_imu_.Acc, last_imu_.Gyro,
                         VB_cur_eig.segment(3, 3), VB_cur_eig.segment(6, 3),
                         last_imu_.acc_n, last_imu_.gyr_n, 
                         last_imu_.acc_w, last_imu_.gyr_w);
        
        } // endif (imu_initialized_)
        
        nb_images_processed_++;
    }

    void GSSlam::optimizationLoop()
    {
        while (!stop_optimization_.load()) {
            {
                std::lock_guard<std::mutex> lock(optimization_mutex_);
                optimizeGaussians(gaussian_iterations_, eta_gaussian_);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    void GSSlam::initializeFirstFrame(const cv::Mat &rgb, const cv::Mat &depth, const CameraPose &odometry_pose)
    {
        // Convertimos odometry_pose (x, y, z, w) a CameraPose interno (w, x, y, z)
        CameraPose odom_pose_converted = odometry_pose;
        odom_pose_converted.orientation = make_float4(
            odometry_pose.orientation.w,
            odometry_pose.orientation.x,
            odometry_pose.orientation.y,
            odometry_pose.orientation.z);

        // Inicializar gaussianas desde RGB-D
        initializeGaussiansFromRgbd(
            rgb,
            depth,
            odom_pose_converted,
            0.001f
        );
        
        // Actualizar pose
        current_pose_ = odom_pose_converted;
        P_cur_[0] = odom_pose_converted.position.x;
        P_cur_[1] = odom_pose_converted.position.y;
        P_cur_[2] = odom_pose_converted.position.z;
        P_cur_[3] = odom_pose_converted.orientation.y;
        P_cur_[4] = odom_pose_converted.orientation.z;
        P_cur_[5] = odom_pose_converted.orientation.w;
        P_cur_[6] = odom_pose_converted.orientation.x;
    }
    
    void GSSlam::rasterizeFill(cv::cuda::GpuMat &rendered_rgb, cv::cuda::GpuMat &rendered_depth)
    {
        if (rendered_rgb_gpu_.empty() || rendered_depth_gpu_.empty()) {
            rendered_rgb.release();
            rendered_depth.release();
            return;
        }

        rendered_rgb = rendered_rgb_gpu_;
        rendered_depth = rendered_depth_gpu_;
    }

    void GSSlam::rasterizeWithErrors(const cv::Mat &rgb_gt, const cv::Mat &depth_gt)
    {
        // TODO: Rasterizar y calcular errores RGB-D en el mismo kernel
        // Este método debe reemplazar computeRenderingErrors() cuando
        // exista un kernel tipo forwardPassTileKernelWithError.
        (void)rgb_gt;
        (void)depth_gt;
    }
    
    void GSSlam::optimizePose(int nb_iterations, float eta)
    {
        // TODO: Optimizar pose usando errores visuales
        // Usar métodos Gauss-Newton o Adam
        // Por ahora solo placeholder
    }
    
    void GSSlam::optimizeGaussians(int nb_iterations, float eta)
    {
        if (n_Gaussians == 0 || keyframes_.empty()) {
            return;
        }

        const int iterations = std::max(1, nb_iterations);
        const size_t total_keyframes = keyframes_.size();

        auto add_unique = [](std::vector<int> &indices, int idx) {
            if (idx < 0) {
                return;
            }
            for (int existing : indices) {
                if (existing == idx) {
                    return;
                }
            }
            indices.push_back(idx);
        };

        for (int it = 0; it < iterations; ++it) {
            std::vector<int> selected;
            selected.reserve(4);

            int last_idx = current_keyframe_idx_;
            if (last_idx < 0 || static_cast<size_t>(last_idx) >= total_keyframes) {
                last_idx = static_cast<int>(total_keyframes - 1);
            }

            // 1) Optimizar los ultimos dos frames
            add_unique(selected, last_idx);
            if (last_idx > 0) {
                add_unique(selected, last_idx - 1);
            }

            // 2) Optimizar dos frames elegidos por KeyframeSelector
            if (selected.size() < 4 && total_keyframes > selected.size()) {
                std::vector<int> sampled;
                try {
                    sampled = keyframe_selector_.sample(2,
                                                        static_cast<int>(total_keyframes),
                                                        gaussian_sampling_method_,
                                                        {});
                } catch (const std::invalid_argument &) {
                    sampled = keyframe_selector_.sample(2,
                                                        static_cast<int>(total_keyframes),
                                                        "beta_binomial",
                                                        {});
                }
                for (int idx : sampled) {
                    add_unique(selected, idx);
                }
            }

            for (int idx : selected) {
                if (idx >= 0 && static_cast<size_t>(idx) < total_keyframes) {
                    optimizeGaussiansKeyframe(keyframes_[static_cast<size_t>(idx)], eta);
                }
            }
        }
    }

    void GSSlam::optimizeGaussiansKeyframe(const KeyframeData &keyframe, float eta)
    {
        if (n_Gaussians == 0 || keyframe.color_img.empty() || keyframe.depth_img.empty()) {
            return;
        }

        // Paso 1: preparar rasterizacion (usa buffers existentes: positions_2d_, inv_covariances_2d_, p_hats_, depths_)
        prepareRasterization(keyframe.getPose(), keyframe.getIntrinsics(),
                             keyframe.getWidth(), keyframe.getHeight());

        if (last_nb_instances_ == 0) {
            return;
        }

        // CRITICAL VALIDATION: Verificar integridad de todos los buffers críticos
        if (positions_2d_.empty() || depths_.empty() || inv_covariances_2d_.empty() ||
            covariances_2d_.empty() || p_hats_.empty() || tile_ranges_.empty() || 
            gaussian_indices_.empty() || tile_counts_.empty() || hashes_.empty()) {
            std::cerr << "ERROR: Buffers vacios tras prepareRasterization()" << std::endl;
            std::cerr << "  positions_2d_: " << positions_2d_.size() << std::endl;
            std::cerr << "  depths_: " << depths_.size() << std::endl;
            std::cerr << "  inv_covariances_2d_: " << inv_covariances_2d_.size() << std::endl;
            std::cerr << "  tile_ranges_: " << tile_ranges_.size() << std::endl;
            return;
        }

        // Verificar que todos los buffers tienen tamaño compatible
        if (positions_2d_.size() != depths_.size() || 
            positions_2d_.size() != inv_covariances_2d_.size()) {
            std::cerr << "ERROR: Size mismatch en buffers de gaussianas: " 
                      << positions_2d_.size() << ", " 
                      << depths_.size() << ", "
                      << inv_covariances_2d_.size() << std::endl;
            return;
        }

        // Paso 2: buckets por tile (prefix sum)
        int num_tiles_total = num_tiles_.x * num_tiles_.y;
        if (num_tiles_total <= 0 || tile_ranges_.empty()) {
            return;
        }

        thrust::device_vector<uint32_t> bucket_offsets(num_tiles_total);
        if (bucket_offsets.empty()) {
            return;
        }
        
        perTileBucketCount<<<(num_tiles_total + 255) / 256, 256>>>(
            thrust::raw_pointer_cast(bucket_offsets.data()),
            thrust::raw_pointer_cast(tile_ranges_.data()),
            num_tiles_total);
        cudaDeviceSynchronize();

        thrust::inclusive_scan(
            bucket_offsets.begin(),
            bucket_offsets.end(),
            bucket_offsets.begin());

        uint32_t num_buckets = 0;
        if (!bucket_offsets.empty()) {
            num_buckets = bucket_offsets.back();
        }
        if (num_buckets == 0) {
            return;
        }

        // Paso 3: buffers auxiliares para forward/backward
        const int width = keyframe.getWidth();
        const int height = keyframe.getHeight();
        const uint32_t num_pixels = static_cast<uint32_t>(width * height);
        const uint32_t block_size = tile_size_.x * tile_size_.y;

        thrust::device_vector<uint32_t> bucket_to_tile(num_buckets);
        thrust::device_vector<float> sampled_T(num_buckets * block_size);
        thrust::device_vector<float3> sampled_ar(num_buckets * block_size);
        thrust::device_vector<float> final_T(num_pixels);
        thrust::device_vector<uint32_t> n_contrib(num_pixels);
        thrust::device_vector<uint32_t> max_contrib(num_tiles_total);
        thrust::device_vector<float3> output_color(num_pixels);
        thrust::device_vector<float> output_depth(num_pixels);
        thrust::device_vector<float3> color_error(num_pixels);
        thrust::device_vector<float> depth_error(num_pixels);

        // Paso 4: acumuladores de gradiente (2D)
        thrust::device_vector<DeltaGaussian2D> delta_gaussians(n_Gaussians);
        cudaMemset(thrust::raw_pointer_cast(delta_gaussians.data()),
                   0,
                   n_Gaussians * sizeof(DeltaGaussian2D));

        // Paso 5: preparar observaciones (float3 RGB + float depth)
        cv::cuda::GpuMat observed_rgb_f;
        if (keyframe.color_img.type() == CV_32FC3) {
            observed_rgb_f = keyframe.color_img;
        } else {
            keyframe.color_img.convertTo(observed_rgb_f, CV_32FC3, 1.0 / 255.0);
        }

        cv::cuda::GpuMat observed_depth_f;
        if (keyframe.depth_img.type() == CV_32FC1) {
            observed_depth_f = keyframe.depth_img;
        } else {
            keyframe.depth_img.convertTo(observed_depth_f, CV_32FC1);
        }

        const float3 *observed_rgb = reinterpret_cast<const float3 *>(
            observed_rgb_f.ptr<cv::Vec3f>());
        const float *observed_depth = observed_depth_f.ptr<float>();

        // Paso 6: forward pass (kernel por tile)
        optimizeGaussiansForwardPass<<<dim3(num_tiles_.x, num_tiles_.y),
                                       dim3(tile_size_.x, tile_size_.y)>>>(
            thrust::raw_pointer_cast(tile_ranges_.data()),
            thrust::raw_pointer_cast(gaussian_indices_.data()),
            thrust::raw_pointer_cast(positions_2d_.data()),
            thrust::raw_pointer_cast(inv_covariances_2d_.data()),
            thrust::raw_pointer_cast(p_hats_.data()),
            thrust::raw_pointer_cast(depths_.data()),
            thrust::raw_pointer_cast(gaussians_.colors.data()),
            thrust::raw_pointer_cast(gaussians_.opacities.data()),
            thrust::raw_pointer_cast(bucket_offsets.data()),
            thrust::raw_pointer_cast(bucket_to_tile.data()),
            thrust::raw_pointer_cast(sampled_T.data()),
            thrust::raw_pointer_cast(sampled_ar.data()),
            thrust::raw_pointer_cast(final_T.data()),
            thrust::raw_pointer_cast(n_contrib.data()),
            thrust::raw_pointer_cast(max_contrib.data()),
            thrust::raw_pointer_cast(output_color.data()),
            thrust::raw_pointer_cast(output_depth.data()),
            thrust::raw_pointer_cast(color_error.data()),
            thrust::raw_pointer_cast(depth_error.data()),
            observed_rgb,
            observed_depth,
            bg_color_,
            num_tiles_,
            width,
            height);

        // Paso 7: backward/accumulation pass (kernel por bucket)
        optimizeGaussiansPerGaussianPass<<<num_buckets, 32>>>(
            thrust::raw_pointer_cast(tile_ranges_.data()),
            thrust::raw_pointer_cast(gaussian_indices_.data()),
            thrust::raw_pointer_cast(positions_2d_.data()),
            thrust::raw_pointer_cast(inv_covariances_2d_.data()),
            thrust::raw_pointer_cast(p_hats_.data()),
            thrust::raw_pointer_cast(depths_.data()),
            thrust::raw_pointer_cast(gaussians_.colors.data()),
            thrust::raw_pointer_cast(gaussians_.opacities.data()),
            thrust::raw_pointer_cast(bucket_offsets.data()),
            thrust::raw_pointer_cast(bucket_to_tile.data()),
            thrust::raw_pointer_cast(sampled_T.data()),
            thrust::raw_pointer_cast(sampled_ar.data()),
            thrust::raw_pointer_cast(n_contrib.data()),
            thrust::raw_pointer_cast(max_contrib.data()),
            thrust::raw_pointer_cast(output_color.data()),
            thrust::raw_pointer_cast(output_depth.data()),
            thrust::raw_pointer_cast(color_error.data()),
            thrust::raw_pointer_cast(depth_error.data()),
            thrust::raw_pointer_cast(delta_gaussians.data()),
            /*w_depth=*/1.0f,
            /*w_dist=*/0.1f,
            num_tiles_,
            width,
            height,
            static_cast<int>(num_buckets));

        // Paso 8: convertir gradientes 2D a 3D (base para Adam)
        thrust::device_vector<DeltaGaussian3D> delta_gaussians_3d(n_Gaussians);
        const float lambda_iso = 0.01f;
        dim3 block(256);
        dim3 grid((n_Gaussians + block.x - 1) / block.x);
        computeDeltaGaussians3D_kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(delta_gaussians_3d.data()),
            thrust::raw_pointer_cast(gaussians_.positions.data()),
            thrust::raw_pointer_cast(gaussians_.scales.data()),
            thrust::raw_pointer_cast(gaussians_.orientations.data()),
            thrust::raw_pointer_cast(gaussians_.colors.data()),
            thrust::raw_pointer_cast(gaussians_.opacities.data()),
            thrust::raw_pointer_cast(delta_gaussians.data()),
            keyframe.getPose(),
            keyframe.getIntrinsics(),
            lambda_iso,
            n_Gaussians);

        // Paso 9: Aplicar actualización Adam a parámetros de gaussianas
        // Inicializamos estado Adam si no existe (una sola vez, se reutiliza)
        if (adam_states_.empty()) {
            adam_states_.resize(n_Gaussians);
            // Inicializar a cero (momentum y varianza) con contador de paso t=0
            cudaMemset(thrust::raw_pointer_cast(adam_states_.data()),
                      0,
                      n_Gaussians * sizeof(AdamStateGaussian3D));
        }

        // Validamos que adam_states tenga el tamaño correcto
        if (adam_states_.size() < n_Gaussians) {
            adam_states_.resize(n_Gaussians);
        }

        // Hiperparámetros de Adam
        const float adam_beta1 = 0.9f;   // Momentum decay
        const float adam_beta2 = 0.999f; // RMSprop decay
        const float adam_eps = 1e-8f;    // Epsilon para estabilidad numérica

        // Ejecutamos kernel de actualización con Adam
        dim3 adam_block(256);
        dim3 adam_grid((n_Gaussians + adam_block.x - 1) / adam_block.x);

        updateGaussiansParametersAdam_kernel<<<adam_grid, adam_block>>>(
            thrust::raw_pointer_cast(gaussians_.positions.data()),
            thrust::raw_pointer_cast(gaussians_.scales.data()),
            thrust::raw_pointer_cast(gaussians_.orientations.data()),
            thrust::raw_pointer_cast(gaussians_.colors.data()),
            thrust::raw_pointer_cast(gaussians_.opacities.data()),
            thrust::raw_pointer_cast(adam_states_.data()),
            thrust::raw_pointer_cast(delta_gaussians_3d.data()),
            eta,
            adam_beta1,
            adam_beta2,
            adam_eps,
            n_Gaussians);

        cudaDeviceSynchronize();
    }
    
    void GSSlam::addKeyframe()
    {
        // Crear keyframe usando KeyframeData
        KeyframeData kf;
        
        // Configurar pose
        kf.setPose(current_pose_);
        
        // Configurar intrinsics
        kf.intrinsics = intrinsics_;
        
        // Copiar imágenes actuales
        if (!pyr_color_.empty() && !pyr_color_[0].empty()) {
            kf.color_img = pyr_color_[0].clone();
        }
        if (!pyr_depth_.empty() && !pyr_depth_[0].empty()) {
            kf.depth_img = pyr_depth_[0].clone();
        }
        
        // Configurar ID y timestamp
        kf.keyframe_id = static_cast<uint32_t>(keyframes_.size());
        kf.timestamp = static_cast<double>(nb_images_processed_);
        
        // Agregar a lista
        keyframes_.push_back(kf);
        keyframe_gaussian_counts_.push_back(n_Gaussians);
        current_keyframe_idx_ = keyframes_.size() - 1;
    }
    
    void GSSlam::densify(const KeyframeData &keyframe)
    {
        if (n_Gaussians == 0) return;
        if (keyframe.color_img.empty() || keyframe.depth_img.empty()) return;

        int width = keyframe.getWidth();
        int height = keyframe.getHeight();

        // Preparamos proyeccion y ranges para calcular el density mask
        prepareRasterization(keyframe.getPose(), keyframe.getIntrinsics(), width, height);

        cv::cuda::GpuMat density_mask(height, width, CV_32FC1);

        dim3 tile_block(tile_size_.x, tile_size_.y);
        dim3 tile_grid(num_tiles_.x, num_tiles_.y);
        computeDensityMask_kernel<<<tile_grid, tile_block>>>(
            density_mask.ptr<float>(),
            thrust::raw_pointer_cast(tile_ranges_.data()),
            thrust::raw_pointer_cast(gaussian_indices_.data()),
            thrust::raw_pointer_cast(positions_2d_.data()),
            thrust::raw_pointer_cast(inv_covariances_2d_.data()),
            thrust::raw_pointer_cast(p_hats_.data()),
            thrust::raw_pointer_cast(gaussians_.colors.data()),
            thrust::raw_pointer_cast(gaussians_.opacities.data()),
            keyframe.depth_img.ptr<float>(),
            thrust::raw_pointer_cast(depths_.data()),
            keyframe.depth_img.step,
            num_tiles_,
            width,
            height,
            density_mask.step);

        uint32_t counter_host = n_Gaussians;
        cudaMemcpy(thrust::raw_pointer_cast(instance_counter_.data()),
                   &counter_host,
                   sizeof(uint32_t),
                   cudaMemcpyHostToDevice);

        uint32_t sample_dx = static_cast<uint32_t>(gauss_init_size_px_);
        uint32_t sample_dy = static_cast<uint32_t>(gauss_init_size_px_);
        int sample_w = (width + sample_dx - 1) / sample_dx;
        int sample_h = (height + sample_dy - 1) / sample_dy;

        dim3 block(16, 16);
        dim3 grid((sample_w + block.x - 1) / block.x,
                  (sample_h + block.y - 1) / block.y);

        densifyGaussians_kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(gaussians_.positions.data()),
            thrust::raw_pointer_cast(gaussians_.scales.data()),
            thrust::raw_pointer_cast(gaussians_.orientations.data()),
            thrust::raw_pointer_cast(gaussians_.colors.data()),
            thrust::raw_pointer_cast(gaussians_.opacities.data()),
            thrust::raw_pointer_cast(instance_counter_.data()),
            keyframe.color_img.ptr<uchar3>(),
            keyframe.color_img.step,
            keyframe.depth_img.ptr<float>(),
            keyframe.depth_img.step,
            nullptr,
            0,
            density_mask.ptr<float>(),
            density_mask.step,
            keyframe.getPose(),
            keyframe.getIntrinsics(),
            sample_dx,
            sample_dy,
            width,
            height,
            max_Gaussians);
        cudaDeviceSynchronize();

        cudaMemcpy(&counter_host,
                   thrust::raw_pointer_cast(instance_counter_.data()),
                   sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);

        n_Gaussians = std::min(counter_host, max_Gaussians);
        std::cout << "Densify: total gaussianas " << n_Gaussians << std::endl;
    }
    
    void GSSlam::prune()
    {
        if (n_Gaussians == 0) return;
        
        // Parámetros de prune
        const float alpha_threshold = 0.05f;       // Opacidad mínima 5%
        const float scale_ratio_threshold = 0.05f; // Ratio mínimo entre escalas
        
        // Preparar buffers
        thrust::device_vector<unsigned char> states(n_Gaussians, 0);
        uint32_t nb_removed_host = 0;
        thrust::device_vector<uint32_t> nb_removed(1, 0);
        
        // Lanzar kernel de prune
        dim3 block(256);
        dim3 grid((n_Gaussians + block.x - 1) / block.x);
        
        pruneGaussians_kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(nb_removed.data()),
            thrust::raw_pointer_cast(states.data()),
            thrust::raw_pointer_cast(gaussians_.scales.data()),
            thrust::raw_pointer_cast(gaussians_.opacities.data()),
            alpha_threshold,
            scale_ratio_threshold,
            n_Gaussians);
        cudaDeviceSynchronize();
        
        // CRITICAL VALIDATION: Verificar integridad antes de ordenar
        if (states.size() != static_cast<size_t>(n_Gaussians)) {
            std::cerr << "ERROR in prune(): states size mismatch!" << std::endl;
            std::cerr << "  states.size(): " << states.size() << std::endl;
            std::cerr << "  n_Gaussians: " << n_Gaussians << std::endl;
            return;
        }
        if (gaussians_.positions.size() < static_cast<size_t>(n_Gaussians)) {
            std::cerr << "ERROR in prune(): positions capacity too small!" << std::endl;
            return;
        }
        
        // Ordenar por estados (0 primero, 0xff al final)
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
            gaussians_.positions.begin(),
            gaussians_.scales.begin(),
            gaussians_.orientations.begin(),
            gaussians_.colors.begin(),
            gaussians_.opacities.begin()));
        
        thrust::sort_by_key(
            states.begin(),
            states.begin() + n_Gaussians,
            zip_begin);
        
        // Copiar número de gaussianas eliminadas
        cudaMemcpy(&nb_removed_host,
                   thrust::raw_pointer_cast(nb_removed.data()),
                   sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        
        // Actualizar contador
        n_Gaussians -= nb_removed_host;
        
        // Mantener capacidad para futuras densificaciones.
        
        if (nb_removed_host > 0) {
            std::cout << "Prune: eliminadas " << nb_removed_host << " gaussianas (quedan: " 
                      << n_Gaussians << ")" << std::endl;
        }
    }
    
    void GSSlam::removeOutliers()
    {
        if (n_Gaussians == 0) return;
        
        // Preparar rasterización para detección de outliers
        int width = pyr_depth_.empty() ? 0 : pyr_depth_[0].cols;
        int height = pyr_depth_.empty() ? 0 : pyr_depth_[0].rows;
        
        if (width == 0 || height == 0) return;
        
        prepareRasterization(current_pose_, intrinsics_, width, height);
        
        // Preparar buffers para outliers
        thrust::device_vector<float> outlier_prob(n_Gaussians, 0.0f);
        thrust::device_vector<float> total_alpha(n_Gaussians, 0.0f);
        thrust::device_vector<unsigned char> states(n_Gaussians, 0);
        
        // Lanzar kernel de detección de outliers
        dim3 block(tile_size_.x, tile_size_.y);
        dim3 grid(num_tiles_.x, num_tiles_.y);

        computeOutliers_kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(outlier_prob.data()),
            thrust::raw_pointer_cast(total_alpha.data()),
            thrust::raw_pointer_cast(tile_ranges_.data()),
            thrust::raw_pointer_cast(gaussian_indices_.data()),
            thrust::raw_pointer_cast(positions_2d_.data()),
            thrust::raw_pointer_cast(inv_covariances_2d_.data()),
            thrust::raw_pointer_cast(p_hats_.data()),
            thrust::raw_pointer_cast(depths_.data()),
            thrust::raw_pointer_cast(gaussians_.opacities.data()),
            pyr_depth_[0].ptr<float>(),
            pyr_depth_[0].step,
            num_tiles_,
            width,
            height);
        cudaDeviceSynchronize();
        
        // Eliminar outliers detectados
        uint32_t nb_removed_host = 0;
        thrust::device_vector<uint32_t> nb_removed(1, 0);
        const float outlier_threshold = 0.6f;  // 60% de contribuciones son outliers
        
        dim3 block2(256);
        dim3 grid2((n_Gaussians + block2.x - 1) / block2.x);
        
        removeOutliers_kernel<<<grid2, block2>>>(
            thrust::raw_pointer_cast(nb_removed.data()),
            thrust::raw_pointer_cast(states.data()),
            thrust::raw_pointer_cast(outlier_prob.data()),
            thrust::raw_pointer_cast(total_alpha.data()),
            outlier_threshold,
            n_Gaussians);
        cudaDeviceSynchronize();
        
        // CRITICAL VALIDATION: Verificar integridad antes de ordenar
        if (states.size() != static_cast<size_t>(n_Gaussians)) {
            std::cerr << "ERROR in removeOutliers(): states size mismatch!" << std::endl;
            std::cerr << "  states.size(): " << states.size() << std::endl;
            std::cerr << "  n_Gaussians: " << n_Gaussians << std::endl;
            return;
        }
        if (gaussians_.positions.size() < static_cast<size_t>(n_Gaussians)) {
            std::cerr << "ERROR in removeOutliers(): positions capacity too small!" << std::endl;
            return;
        }
        
        // Ordenar por estados (0 primero, 0xff al final)
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
            gaussians_.positions.begin(),
            gaussians_.scales.begin(),
            gaussians_.orientations.begin(),
            gaussians_.colors.begin(),
            gaussians_.opacities.begin()));
        
        thrust::sort_by_key(
            states.begin(),
            states.begin() + n_Gaussians,
            zip_begin);
        
        // Copiar número de gaussianas eliminadas
        cudaMemcpy(&nb_removed_host,
                   thrust::raw_pointer_cast(nb_removed.data()),
                   sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        
        // Actualizar contador
        n_Gaussians -= nb_removed_host;
        
        // Mantener capacidad para futuras densificaciones.
        
        if (nb_removed_host > 0) {
            std::cout << "RemoveOutliers: eliminadas " << nb_removed_host << " gaussianas (quedan: " 
                      << n_Gaussians << ")" << std::endl;
        }
    }
    
    float GSSlam::computeCovisibilityRatio()
    {
        if (n_Gaussians == 0 || keyframes_.empty() || pyr_color_.empty()) {
            return 1.0f;
        }

        size_t keyframe_idx = static_cast<size_t>(current_keyframe_idx_);
        if (current_keyframe_idx_ < 0 || keyframe_idx >= keyframes_.size()) {
            keyframe_idx = keyframes_.size() - 1;
        }
        const KeyframeData &keyframe = keyframes_[keyframe_idx];
        if (keyframe.color_img.empty() || keyframe.depth_img.empty()) {
            return 1.0f;
        }

        const int frame_width = pyr_color_[0].cols;
        const int frame_height = pyr_color_[0].rows;
        if (frame_width <= 0 || frame_height <= 0) {
            return 1.0f;
        }

        // Host
        uint32_t h_visUnion = 0;
        uint32_t h_visInter = 0;

        // Device
        uint32_t *d_visUnion = nullptr;
        uint32_t *d_visInter = nullptr;
        unsigned char *d_keyframeVis = nullptr;
        unsigned char *d_frameVis = nullptr;

        // Alocamos memoria en device
        cudaMalloc(&d_visUnion, sizeof(uint32_t));
        cudaMalloc(&d_visInter, sizeof(uint32_t));
        cudaMalloc(&d_keyframeVis, n_Gaussians * sizeof(unsigned char));
        cudaMalloc(&d_frameVis, n_Gaussians * sizeof(unsigned char));

        // Llevamos los contadores al device
        cudaMemcpy(d_visUnion, &h_visUnion, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_visInter, &h_visInter, sizeof(uint32_t), cudaMemcpyHostToDevice);

        // Inicializamos visibilidades a 0
        cudaMemset(d_keyframeVis, 0, n_Gaussians * sizeof(unsigned char));
        cudaMemset(d_frameVis, 0, n_Gaussians * sizeof(unsigned char));

        // Calculamos para el ultimo keyframe
        prepareRasterization(keyframe.getPose(), keyframe.getIntrinsics(),
                             keyframe.getWidth(), keyframe.getHeight());
        if (last_nb_instances_ == 0) {
            cudaFree(d_visUnion);
            cudaFree(d_visInter);
            cudaFree(d_keyframeVis);
            cudaFree(d_frameVis);
            return 0.0f;
        }

        // Validar que los buffers tienen datos válidos
        if (positions_2d_.empty() || depths_.empty() || positions_2d_.size() != depths_.size()) {
            cudaFree(d_visUnion);
            cudaFree(d_visInter);
            cudaFree(d_keyframeVis);
            cudaFree(d_frameVis);
            return 0.0f;
        }

        // Crear img_positions con el tamaño correcto (solo Gaussianas activas)
        uint32_t valid_size = std::min(positions_2d_.size(), depths_.size());
        if (valid_size == 0 || valid_size != inv_covariances_2d_.size() || gaussians_.opacities.size() < valid_size) {
            // Si hay inconsistencia de tamaños, retornar
            cudaFree(d_visUnion);
            cudaFree(d_visInter);
            cudaFree(d_keyframeVis);
            cudaFree(d_frameVis);
            return 0.0f;
        }

        thrust::device_vector<float3> img_positions(valid_size);
        
        // Crear zip iterator solo para elementos válidos
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(
            positions_2d_.begin(), 
            depths_.begin()
        ));
        auto end = thrust::make_zip_iterator(thrust::make_tuple(
            positions_2d_.begin() + valid_size, 
            depths_.begin() + valid_size
        ));
        thrust::transform(begin, end, img_positions.begin(), ToFloat3());

        computeGaussiansVisibility_kernel<<<
            dim3(num_tiles_.x, num_tiles_.y),
            dim3(tile_size_.x, tile_size_.y)
        >>>(
            d_keyframeVis,
            thrust::raw_pointer_cast(tile_ranges_.data()),
            thrust::raw_pointer_cast(gaussian_indices_.data()),
            thrust::raw_pointer_cast(img_positions.data()),
            thrust::raw_pointer_cast(inv_covariances_2d_.data()),
            thrust::raw_pointer_cast(gaussians_.opacities.data()),
            num_tiles_,
            keyframe.getWidth(),
            keyframe.getHeight()
        );

        // Calculamos para el frame actual
        prepareRasterization(current_pose_, intrinsics_, frame_width, frame_height);
        if (last_nb_instances_ == 0) {
            cudaFree(d_visUnion);
            cudaFree(d_visInter);
            cudaFree(d_keyframeVis);
            cudaFree(d_frameVis);
            return 0.0f;
        }

        // Validar coherencia de buffers para el frame actual
        if (positions_2d_.size() != depths_.size() || positions_2d_.size() != inv_covariances_2d_.size()) {
            cudaFree(d_visUnion);
            cudaFree(d_visInter);
            cudaFree(d_keyframeVis);
            cudaFree(d_frameVis);
            return 0.0f;
        }

        // Rellenar img_positions para frame actual
        uint32_t frame_valid_size = std::min(positions_2d_.size(), depths_.size());
        if (frame_valid_size == 0 || frame_valid_size != inv_covariances_2d_.size() ||
            gaussians_.opacities.size() < frame_valid_size) {
            cudaFree(d_visUnion);
            cudaFree(d_visInter);
            cudaFree(d_keyframeVis);
            cudaFree(d_frameVis);
            return 0.0f;
        }

        if (img_positions.size() < frame_valid_size) {
            img_positions.resize(frame_valid_size);
        }
        auto frame_begin = thrust::make_zip_iterator(thrust::make_tuple(
            positions_2d_.begin(), 
            depths_.begin()
        ));
        auto frame_end = thrust::make_zip_iterator(thrust::make_tuple(
            positions_2d_.begin() + frame_valid_size, 
            depths_.begin() + frame_valid_size
        ));
        thrust::transform(frame_begin, frame_end, img_positions.begin(), ToFloat3());

        computeGaussiansVisibility_kernel<<<
            dim3(num_tiles_.x, num_tiles_.y),
            dim3(tile_size_.x, tile_size_.y)
        >>>(
            d_frameVis,
            thrust::raw_pointer_cast(tile_ranges_.data()),
            thrust::raw_pointer_cast(gaussian_indices_.data()),
            thrust::raw_pointer_cast(img_positions.data()),
            thrust::raw_pointer_cast(inv_covariances_2d_.data()),
            thrust::raw_pointer_cast(gaussians_.opacities.data()),
            num_tiles_,
            frame_width,
            frame_height
        );

        // Calculamos la covisibilidad
        const int covis_block_size = 256;
        computeGaussiansCovisibility_kernel<<<
            (n_Gaussians + covis_block_size - 1) / covis_block_size,
            covis_block_size
        >>>(
            d_visInter,
            d_visUnion,
            d_keyframeVis,
            d_frameVis,
            n_Gaussians
        );

        // Copiamos resultados al host
        cudaMemcpy(&h_visInter, d_visInter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_visUnion, d_visUnion, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Liberamos memoria
        cudaFree(d_visUnion);
        cudaFree(d_visInter);
        cudaFree(d_keyframeVis);
        cudaFree(d_frameVis);

        // Devolvemos el ratio

        if (h_visUnion == 0)
            return 0.0f; // Caso borde de division por cero

        return h_visInter / static_cast<float>(h_visUnion);
    }
    
    void GSSlam::initWarping(const CameraPose &camera_pose)
    {
        // TODO: Inicializar warping para tracking
        // Usado en métodos de pose estimation con warping
        (void)camera_pose;
    }
    
    void GSSlam::optimizePoseMultiScale()
    {
        // TODO: Optimización multi-escala completa
        // Wrapper que llama optimizePose en cada nivel de pirámide
    }
    
    void GSSlam::initAndCopyImgs(const cv::Mat &rgb, const cv::Mat &depth)
    {
        if (rgb.empty() || depth.empty()) return;
        
        // Convertir RGB a formato correcto
        cv::Mat rgb_bgr;
        if (rgb.type() == CV_8UC3) {
            rgb_bgr = rgb;
        } else {
            rgb.convertTo(rgb_bgr, CV_8UC3);
        }
        
        // Convertir depth a float
        cv::Mat depth_float;
        if (depth.type() == CV_32FC1) {
            depth_float = depth;
        } else if (depth.type() == CV_16UC1) {
            depth.convertTo(depth_float, CV_32FC1, 0.001f);
        } else {
            return;
        }
        
        std::vector<cv::Mat> pyr_color_cpu(nb_pyr_levels_);
        std::vector<cv::Mat> pyr_depth_cpu(nb_pyr_levels_);

        pyr_color_cpu[0] = rgb_bgr;
        pyr_depth_cpu[0] = depth_float;

        for (int i = 1; i < nb_pyr_levels_; i++) {
            cv::pyrDown(pyr_color_cpu[i - 1], pyr_color_cpu[i]);
            cv::pyrDown(pyr_depth_cpu[i - 1], pyr_depth_cpu[i]);
        }

        for (int i = 0; i < nb_pyr_levels_; i++) {
            // Convertir color a float para Sobel
            cv::Mat pyr_color_float;
            printf("DEBUG: pyr_color_cpu[%d] type=%d (should be %d=CV_8UC3), converting to float\n", 
                   i, pyr_color_cpu[i].type(), CV_8UC3);
            pyr_color_cpu[i].convertTo(pyr_color_float, CV_32FC3, 1.0/255.0);
            printf("DEBUG: pyr_color_float type=%d (should be %d=CV_32FC3)\n", 
                   pyr_color_float.type(), CV_32FC3);
            
            pyr_color_[i].upload(pyr_color_float);
            pyr_depth_[i].upload(pyr_depth_cpu[i]);

            cv::Mat dx_cpu;
            cv::Mat dy_cpu;
            cv::Sobel(pyr_color_float, dx_cpu, CV_32F, 1, 0, 3);
            cv::Sobel(pyr_color_float, dy_cpu, CV_32F, 0, 1, 3);
            pyr_dx_[i].upload(dx_cpu);
            pyr_dy_[i].upload(dy_cpu);
        }
        
        // Actualizar referencias legacy
        rgb_gpu_ = pyr_color_[0];
        depth_gpu_ = pyr_depth_[0];
    }
    
    void GSSlam::computeRenderingErrors(const cv::Mat &rgb_gt, const cv::Mat &depth_gt)
    {
        if (rgb_gt.empty() || depth_gt.empty()) return;
        if (rendered_rgb_gpu_.empty() || rendered_depth_gpu_.empty()) return;

        cv::Mat rgb_bgr;
        if (rgb_gt.type() == CV_8UC3) {
            rgb_bgr = rgb_gt;
        } else {
            rgb_gt.convertTo(rgb_bgr, CV_8UC3);
        }

        cv::Mat depth_float;
        if (depth_gt.type() == CV_32FC1) {
            depth_float = depth_gt;
        } else if (depth_gt.type() == CV_16UC1) {
            depth_gt.convertTo(depth_float, CV_32FC1, 0.001f);
        } else {
            return;
        }

        cv::Mat rendered_rgb_cpu;
        cv::Mat rendered_depth_cpu;
        rendered_rgb_gpu_.download(rendered_rgb_cpu);
        rendered_depth_gpu_.download(rendered_depth_cpu);

        cv::Mat rendered_rgb_f;
        cv::Mat rgb_gt_f;
        rendered_rgb_cpu.convertTo(rendered_rgb_f, CV_32FC3, 1.0 / 255.0);
        rgb_bgr.convertTo(rgb_gt_f, CV_32FC3, 1.0 / 255.0);

        cv::Mat residual_rgb_f;
        cv::subtract(rendered_rgb_f, rgb_gt_f, residual_rgb_f);

        cv::Mat residual_rgb_gray_f;
        cv::cvtColor(residual_rgb_f, residual_rgb_gray_f, cv::COLOR_BGR2GRAY);

        cv::Mat residual_depth_f;
        cv::subtract(rendered_depth_cpu, depth_float, residual_depth_f);

        const double depth_weight = 0.1;
        cv::Mat error_map_cpu;
        cv::addWeighted(residual_rgb_gray_f, 1.0, residual_depth_f, depth_weight, 0.0, error_map_cpu);

        error_map_gpu_.upload(error_map_cpu);
    }
    
    void GSSlam::addImuMeasurement(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyro)
    {
        if (!preint_ || !preint_->is_initialized) {
            return;
        }
        
        // Agregar medición a la preintegración
        preint_->add_imu(dt, acc, gyro);
        
        // Guardar última medición
        last_imu_.Acc = acc;
        last_imu_.Gyro = gyro;
    }
    
    void GSSlam::initializeImu(const ImuData &imu_data)
    {
        if (!preint_) {
            return;
        }
        
        // Guardar datos del IMU
        last_imu_ = imu_data;
        
        // Inicializar biases a cero (serán estimados durante optimización)
        Eigen::Vector3d ba = Eigen::Vector3d::Zero();
        Eigen::Vector3d bg = Eigen::Vector3d::Zero();
        
        // Inicializar preintegración
        preint_->init(imu_data.Acc, imu_data.Gyro, ba, bg,
                     imu_data.acc_n, imu_data.gyr_n,
                     imu_data.acc_w, imu_data.gyr_w);
        
        // Marcar como inicializada
        imu_initialized_ = true;
    }

    void GSSlam::computeRgbdPoseJacobians(Eigen::Matrix<double, 6, 6> &JtJ,
                                          Eigen::Vector<double, 6> &Jtr,
                                          int level,
                                          const Eigen::Vector3d &P_imu,
                                          const Eigen::Quaterniond &Q_imu)
    {
        // ============================================================
        // PASO 1: Transformar pose IMU → pose cámara
        // ============================================================
        // Aplicar extrínsecos IMU->cámara
        Eigen::Vector3d P_cam = P_imu + Q_imu * t_imu_cam_;
        Eigen::Quaterniond Q_cam = Q_imu * q_imu_cam_;
        CameraPose camera_pose;
        camera_pose.position = make_float3(static_cast<float>(P_cam.x()),
                           static_cast<float>(P_cam.y()),
                           static_cast<float>(P_cam.z()));
        
        Eigen::Quaternionf Q_cam_f = Q_cam.cast<float>();
        camera_pose.orientation = make_float4(Q_cam_f.w(), Q_cam_f.x(), Q_cam_f.y(), Q_cam_f.z());

        // ============================================================
        // PASO 2: Preparar rasterización (proyectar gaussianas)
        // ============================================================
        int pyr_level = std::min(level, nb_pyr_levels_ - 1);
        
        // Obtener dimensiones del nivel de pirámide
        int width = pyr_color_[pyr_level].cols;
        int height = pyr_color_[pyr_level].rows;
        
        // Escalar intrínsecos según nivel
        IntrinsicParameters level_intrinsics = intrinsics_;
        float scale = 1.0f / (1 << pyr_level);  // 1, 0.5, 0.25, ...
        level_intrinsics.f.x *= scale;
        level_intrinsics.f.y *= scale;
        level_intrinsics.c.x *= scale;
        level_intrinsics.c.y *= scale;

        prepareRasterization(camera_pose, level_intrinsics, width, height);

        // ============================================================
        // PASO 3: Calcular JtJ y Jtr mediante kernel GPU
        // ============================================================
        // Inicializar a cero
        JtJ.setZero();
        Jtr.setZero();

        if (n_Gaussians == 0) {
            return;
        }

        if (pyr_color_.empty() || pyr_depth_.empty()) {
            return;
        }

        cv::cuda::GpuMat observed_rgb_f;
        pyr_color_[pyr_level].convertTo(observed_rgb_f, CV_32FC3, 1.0 / 255.0);

        int num_tiles_x = num_tiles_.x;
        int num_tiles_y = num_tiles_.y;
        int num_tiles_total = num_tiles_x * num_tiles_y;
        if (num_tiles_total == 0) {
            return;
        }

        thrust::device_vector<PoseOptimizationRgbdData> posedata(num_tiles_total);
        cudaMemset(thrust::raw_pointer_cast(posedata.data()),
                   0,
                   sizeof(PoseOptimizationRgbdData) * posedata.size());

        dim3 block(tile_size_.x, tile_size_.y);
        dim3 grid(num_tiles_x, num_tiles_y);

        const float alpha_thresh = 0.2f;
        const float color_thresh = 0.1f;
        const float depth_thresh = 0.2f;

        getRgbdPoseJacobians<<<grid, block>>>(
            thrust::raw_pointer_cast(posedata.data()),
            thrust::raw_pointer_cast(tile_ranges_.data()),
            thrust::raw_pointer_cast(gaussian_indices_.data()),
            thrust::raw_pointer_cast(positions_2d_.data()),
            thrust::raw_pointer_cast(depths_.data()),
            thrust::raw_pointer_cast(inv_covariances_2d_.data()),
            thrust::raw_pointer_cast(p_hats_.data()),
            thrust::raw_pointer_cast(gaussians_.colors.data()),
            thrust::raw_pointer_cast(gaussians_.opacities.data()),
            reinterpret_cast<float3 *>(pyr_dx_[pyr_level].ptr<cv::Vec3f>()),
            reinterpret_cast<float3 *>(pyr_dy_[pyr_level].ptr<cv::Vec3f>()),
            reinterpret_cast<float3 *>(observed_rgb_f.ptr<cv::Vec3f>()),
            pyr_depth_[pyr_level].ptr<float>(),
            camera_pose,
            level_intrinsics,
            bg_color_,
            alpha_thresh,
            color_thresh,
            depth_thresh,
            width,
            height,
            num_tiles_x,
            num_tiles_y);
        cudaDeviceSynchronize();

        thrust::host_vector<PoseOptimizationRgbdData> h_posedata = posedata;
        PoseOptimizationRgbdData total;
        for (int i = 0; i < 6; ++i) {
            total.Jtr[i] = 0.0f;
        }
        for (int i = 0; i < 21; ++i) {
            total.JtJ[i] = 0.0f;
        }

        for (const auto &tile : h_posedata) {
            total += tile;
        }

        int idx = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 6; ++j) {
                double value = static_cast<double>(total.JtJ[idx++]);
                JtJ(i, j) = value;
                JtJ(j, i) = value;
            }
            Jtr(i) = -static_cast<double>(total.Jtr[i]);
        }
        
        // ============================================================
        // PASO 4: Transformar jacobiano de espacio cámara a espacio IMU
        // ============================================================
        // J_imu = J_cam_imu^T * J_cam
        // donde J_cam_imu es el jacobiano de la transformación extrínseca
        Eigen::Matrix3d R_imu_cam = q_imu_cam_.toRotationMatrix();
        Eigen::Matrix3d R_imu = Q_imu.toRotationMatrix();
        Eigen::Matrix3d P_imu_cam_skew = skewSymmetric(t_imu_cam_);

        Eigen::Matrix<double, 6, 6> J_cam_imu = Eigen::Matrix<double, 6, 6>::Zero();
        J_cam_imu.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_cam_imu.block<3, 3>(0, 3) = -R_imu * P_imu_cam_skew;
        J_cam_imu.block<3, 3>(3, 3) = R_imu_cam.transpose();

        JtJ = J_cam_imu.transpose() * JtJ * J_cam_imu;
        Jtr = J_cam_imu.transpose() * Jtr;
    }

    void GSSlam::optimizeWithCeres(int min_pyr_level, int max_iterations)
    {
        // ============================================================
        // PASO 1: Activar residual IMU si ya hay preintegración
        // ============================================================
        if (preint_ && preint_->is_initialized && !imu_residual_added_) {
            imu_cost_ = new ImuCostFunction(preint_shared_);
            problem_.AddResidualBlock(imu_cost_, nullptr,
                                      P_prev_, VB_prev_, P_cur_, VB_cur_);
            imu_residual_added_ = true;
        }

        // ============================================================
        // PASO 2: Configurar solver
        // ============================================================
        options_.max_num_iterations = (max_iterations > 0) ? max_iterations : pose_iterations_;
        
        // ============================================================
        // PASO 4: Optimización multi-nivel (coarse-to-fine)
        // ============================================================
        // Similar a VIGS-Fusion: optimizar en pirámide de imágenes
        // Niveles superiores (baja resolución) primero para robustez
        
        min_pyr_level = std::max(0, std::min(min_pyr_level, nb_pyr_levels_ - 1));

        for (int level = nb_pyr_levels_ - 1; level >= min_pyr_level; level--)
        {
            // Actualizar nivel de pirámide en función de costo visual
            visual_cost_->update(level);
            
            // Resolver problema
            ceres::Solve(options_, &problem_, &summary_);
            
            // Verificar convergencia
            if (summary_.termination_type == ceres::CONVERGENCE)
            {
                // Éxito - convergencia alcanzada
                std::cerr << "INFO: Ceres converged at level " << level 
                    << " in " << summary_.iterations.size() << " iterations" << std::endl;
            }
            else if (summary_.termination_type == ceres::NO_CONVERGENCE)
            {
                // Warning - sin convergencia pero puede seguir
                std::cerr << "WARNING: Ceres NO_CONVERGENCE at level " << level 
                    << ". Iterations: " << summary_.iterations.size() 
                    << ", Final cost: " << summary_.final_cost << std::endl;
            }
            else if (summary_.termination_type == ceres::FAILURE)
            {
                std::cerr << "ERROR: Ceres FAILURE at level " << level 
                    << ": " << summary_.message << std::endl;
                break;
            }
        }
        
        // ============================================================
        // PASO 5: Extraer solución optimizada
        // ============================================================
        // Los valores P_cur_ y VB_cur_ ya están actualizados in-place por Ceres
        
        // Actualizar pose de cámara para renderización
        updateCameraPoseFromImu();
        
        // Logging opcional
        if (nb_images_processed_ % 10 == 0) {
            std::cout << "Ceres summary: "
                      << "iterations=" << summary_.iterations.size()
                      << ", final_cost=" << summary_.final_cost
                      << ", termination=" << summary_.termination_type
                      << std::endl;
        }
    }

} // namespace f_vigs_slam