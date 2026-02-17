#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <thrust/device_vector.h>
#include "f_vigs_slam/GSCudaKernels.cuh"
#include "f_vigs_slam/KeyframeSelector.hpp"
#include "f_vigs_slam/RepresentationClasses.hpp"
#include "f_vigs_slam/Preintegration.hpp"
#include "f_vigs_slam/MarginalizationFactor.hpp"
#include <ceres/ceres.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

// Definimos la clase que se encarga del codigo principal a optimizar en CUDA

namespace f_vigs_slam
{
    class RgbdPoseCostFunction;
    class ImuCostFunction;

    class GSSlam
    {
    public:
        GSSlam();
        ~GSSlam();

        void setIntrinsics(const IntrinsicParameters &params);
        void setGaussInitSizePx(int size_px);
        void setGaussInitScale(float scale);
        void setImuToCamExtrinsics(const Eigen::Vector3d &t_imu_cam,
                       const Eigen::Quaterniond &q_imu_cam);
        
        // Setters para parámetros de optimización
        inline void setPoseIterations(int it) { pose_iterations_ = std::max(1, it); }
        inline void setGaussianIterations(int it) { gaussian_iterations_ = std::max(1, it); }
        inline void setEtaPose(float eta) { eta_pose_ = std::max(1e-5f, eta); }
        inline void setEtaGaussian(float eta) { eta_gaussian_ = std::max(1e-5f, eta); }
        inline void setGaussianSamplingMethod(const std::string &method) { gaussian_sampling_method_ = method; }

        void initializeGaussiansFromRgbd(const cv::Mat &rgb,
                                         const cv::Mat &depth,
                                         const CameraPose &cameraPose,
                                         float depth_scale = 0.001f);

        bool hasGaussians() const;
        uint32_t getGaussiansCount() const;

        // ===== CORE SLAM =====
        // Procesa un frame RGB-D completo: inicializa, renderiza, optimiza
        void compute(const cv::Mat &rgb, const cv::Mat &depth, const CameraPose &odometry_pose);
        
        // Renderización visual
        void prepareRasterization(const CameraPose &camera_pose, const IntrinsicParameters &intrinsics,
                                  int width, int height);
        void rasterize(const CameraPose &camera_pose, const IntrinsicParameters &intrinsics,
                       int width, int height);
        void rasterizeWithErrors(const cv::Mat &rgb_gt, const cv::Mat &depth_gt);
        void rasterizeFill(cv::cuda::GpuMat &rendered_rgb, cv::cuda::GpuMat &rendered_depth);

        bool renderView(const CameraPose &camera_pose,
                const IntrinsicParameters &intrinsics,
                int width, int height,
                cv::cuda::GpuMat &rendered_rgb,
                cv::cuda::GpuMat &rendered_depth);
        
        // Optimización de pose visual multi-escala
        void optimizePose(int nb_iterations, float eta = 0.01f);
        void optimizePoseMultiScale();
        
        // Optimización de gaussianas
        void optimizeGaussians(int nb_iterations, float eta = 0.002f);
        void optimizeGaussiansKeyframe(const KeyframeData &keyframe, float eta = 0.002f);

        void optimizationLoop();
        
        // Keyframes
        void addKeyframe();
        float computeCovisibilityRatio();
        
        // Densificación y pruning
        void densify(const KeyframeData &keyframe);
        void prune();
        void removeOutliers();
        
        // Inicialización y copia de imágenes
        void initAndCopyImgs(const cv::Mat &rgb, const cv::Mat &depth);
        
        // Getters para odometría
        inline const CameraPose& getCameraPose() const { return current_pose_; }
        inline const double* getImuPose() const { return P_cur_; }
        inline const double* getImuVelocity() const { return VB_cur_; }
        inline bool hasIntrinsics() const { return intrinsics_set_; }
        inline bool getIsInitialized() const { return isInitialized; }
        inline int getImageWidth() const { return pyr_color_.empty() ? 0 : pyr_color_[0].cols; }
        inline int getImageHeight() const { return pyr_color_.empty() ? 0 : pyr_color_[0].rows; }
        
        // ===== IMU METHODS =====
        /**
         * @brief Agrega medición IMU a la preintegración
         * @param dt Delta tiempo desde última medición (segundos)
         * @param acc Aceleración lineal medida (m/s²)
         * @param gyro Velocidad angular medida (rad/s)
         */
        void addImuMeasurement(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyro);
        
        /**
         * @brief Inicializa IMU con parámetros y biases iniciales
         * @param imu_data Estructura con parámetros del IMU
         */
        void initializeImu(const ImuData &imu_data);

        /**
         * @brief Optimización visual-inertial con Ceres Solver (Multi-factor)
         * 
         * Construye problema de optimización combinando:
         * - **Factor Visual**: RgbdPoseCost (residual renderizado vs observado)
         * - **Factor IMU**: ImuCostFunction (preintegración entre P_prev/VB_prev → P_cur/VB_cur)
         * - **Factor de Marginalización**: MarginalizationFactor (información marginalizada de keyframes viejos)
         * 
         * **Parámetros optimizados:**
         * - P_prev, VB_prev: pose anterior y velocidad+biases
         * - P_cur, VB_cur: pose actual y velocidad+biases
         * 
         * **Parameterización:**
         * - PoseLocalParameterization para SE(3) (7D global → 6D local)
         * - Espacio lineal para velocity+biases (9D)
         * 
         * **Configuración del Solver:**
         * - Solver lineal: DENSE_QR
         * - Algoritmo: Levenberg-Marquardt (robusto)
         * - Iteraciones: pose_iterations_
         * - Tolerancia: 1e-6 (convergencia)
         * 
         * Extrae solución y actualiza P_cur_, VB_cur_ con valores optimizados
         */
        void optimizeWithCeres(int min_pyr_level = 0, int max_iterations = -1);

        /**
         * @brief Calcula jacobianos visuales para optimización de pose
         * 
         * Este método es llamado por RgbdPoseCostFunction para obtener
         * la matriz Hessiana aproximada (J^T * J) y el gradiente (J^T * r).
         * 
         * **Proceso:**
         * 1. Transforma pose IMU → pose cámara
         * 2. Proyecta gaussianas 3D a espacio pantalla
         * 3. Rasteriza para obtener imagen renderizada
         * 4. Calcula residuales RGB-D píxel a píxel
         * 5. Acumula J^T*J y J^T*r mediante kernels GPU
         * 6. Transforma jacobianos de espacio cámara a espacio IMU
         * 
         * @param JtJ Output: Hessiana aproximada 6×6 (traslación + rotación)
         * @param Jtr Output: Gradiente 6×1
         * @param level Nivel de pirámide (0=alta res, N-1=baja res)
         * @param P_imu Posición IMU [x, y, z]
         * @param Q_imu Orientación IMU (quaternion unitario)
         */
        void computeRgbdPoseJacobians(Eigen::Matrix<double, 6, 6> &JtJ,
                          Eigen::Vector<double, 6> &Jtr,
                          int level,
                          const Eigen::Vector3d &P_imu,
                          const Eigen::Quaterniond &Q_imu);

    private:
        void ensureCapacity(uint32_t required);
        void initializeFirstFrame(const cv::Mat &rgb, const cv::Mat &depth, const CameraPose &odometry_pose);
        void computeRenderingErrors(const cv::Mat &rgb_gt, const cv::Mat &depth_gt);
        void initWarping(const CameraPose &camera_pose);
        void updateCameraPoseFromImu();

        // ===== GAUSSIANAS Y DATOS =====
        Gaussians gaussians_;
        thrust::device_vector<uint32_t> instance_counter_;
        thrust::device_vector<uint32_t> instance_counter_screen_;

        // ===== PARAMETROS INTRÍNSECOS =====
        IntrinsicParameters intrinsics_;
        bool intrinsics_set_ = false;

        // ===== ESTADO DE GAUSSIANAS =====
        uint32_t n_Gaussians;
        uint32_t max_Gaussians;
        bool isFirstImage;
        bool isInitialized;

        int gauss_init_size_px_ = 7;
        float gauss_init_scale_ = 0.01f;
        float gauss_init_opacity_ = 0.8f;

        // ===== IMAGENES EN GPU =====
        cv::cuda::GpuMat rgb_gpu_;
        cv::cuda::GpuMat depth_gpu_;
        cv::cuda::GpuMat rendered_rgb_gpu_;
        cv::cuda::GpuMat rendered_depth_gpu_;
        cv::cuda::GpuMat error_map_gpu_;  // Mapa de errores RGB-D

        // ===== SCREEN-SPACE GAUSSIANS (2D projection) =====
        thrust::device_vector<float2> positions_2d_;     // Posiciones proyectadas en pantalla (x, y)
        thrust::device_vector<float3> covariances_2d_;   // Covarianzas 2D (σ_xx, σ_yy, σ_xy)
        thrust::device_vector<float3> inv_covariances_2d_; // Inversas 2D (σ_xx, σ_yy, σ_xy)
        thrust::device_vector<float2> p_hats_;           // Gradiente de profundidad en pixel (dp/dx, dp/dy)
        thrust::device_vector<float> depths_;            // Profundidades Z en cámara
        
        // ===== TILE COUNTING =====
        thrust::device_vector<uint32_t> tile_counts_;    // Número de tiles cubiertos por cada gaussiana
        thrust::device_vector<uint32_t> tile_offsets_;   // Offsets acumulativos (para scatter write)
        
        // ===== TILE SORTING (para depth ordering) =====
        thrust::device_vector<uint64_t> hashes_;         // Hash: (tileID << 32) | depth_as_uint
        thrust::device_vector<uint32_t> gaussian_indices_; // Indices de gaussianas ordenadas
        thrust::device_vector<uint2> tile_ranges_;       // Rangos [start, end] por tile
        uint32_t last_nb_instances_;                     // Número total de hashes en última rasterización
        
        // ===== TILE CONFIGURATION =====
        uint2 tile_size_;    // Tamaño de tile en pixels (ej: 16x16)
        uint2 num_tiles_;    // Número de tiles (width/16, height/16)
        float3 bg_color_;    // Color de fondo

        // ===== GRADIENTES Y DATOS DE OPTIMIZACIÓN =====
        thrust::device_vector<float3> gaussian_gradients_;  // Gradientes de posición
        thrust::device_vector<float> opacity_gradients_;    // Gradientes de opacidad
        thrust::device_vector<AdamStateGaussian3D> adam_states_;  // Estado Adam persistente por gaussiana
        
        // ===== ESTADO IMU =====
        bool imu_initialized_ = false;  // Flag de inicialización IMU
        Preintegration* preint_ = nullptr;  // Preintegración IMU
        ImuData last_imu_;  // Última medición IMU
        Eigen::Quaterniond q_imu_cam_ = Eigen::Quaterniond::Identity();
        Eigen::Vector3d t_imu_cam_ = Eigen::Vector3d::Zero();
        
        // Pose: [x, y, z, qx, qy, qz, qw]
        double P_cur_[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
        double P_prev_[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
        // Velocidad y biases: [vx, vy, vz, ba_x, ba_y, ba_z, bg_x, bg_y, bg_z]
        double VB_cur_[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double VB_prev_[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        
        // Pose actual de la cámara
        CameraPose current_pose_;
        
        // ===== PARÁMETROS DE OPTIMIZACIÓN =====
        int pose_iterations_ = 4;
        int gaussian_iterations_ = 10;
        float eta_pose_ = 0.01f;
        float eta_gaussian_ = 0.002f;
        std::string gaussian_sampling_method_ = "beta_binomial";

        // ===== CERES (reutilizable) =====
        ceres::Problem problem_;
        ceres::Solver::Options options_;
        ceres::Solver::Summary summary_;
        RgbdPoseCostFunction* visual_cost_ = nullptr;
        ImuCostFunction* imu_cost_ = nullptr;
        std::shared_ptr<Preintegration> preint_shared_;
        bool imu_residual_added_ = false;

        // ===== MARGINALIZACIÓN =====
        MarginalizationInfo marginalization_info_;
        MarginalizationFactor* marginalization_cost_ = nullptr;
        
        // ===== PIRÁMIDES MULTI-ESCALA =====
        int nb_pyr_levels_ = 3;
        std::vector<cv::cuda::GpuMat> pyr_color_;
        std::vector<cv::cuda::GpuMat> pyr_depth_;
        std::vector<cv::cuda::GpuMat> pyr_dx_;
        std::vector<cv::cuda::GpuMat> pyr_dy_;
        
        // ===== KEYFRAMES =====
        std::vector<KeyframeData> keyframes_;
        KeyframeSelector keyframe_selector_;
        std::vector<uint32_t> keyframe_gaussian_counts_;  // Número de gaussianas por keyframe
        int current_keyframe_idx_ = 0;
        float covisibility_threshold_ = 0.6f;
        
        // ===== CONTADORES =====
        int nb_images_processed_ = 0;
        bool first_image_ = true;

        // ===== OPTIMIZATION THREAD =====
        std::thread optimize_thread_;
        std::atomic<bool> stop_optimization_{false};
        std::mutex optimization_mutex_;
    };
}
