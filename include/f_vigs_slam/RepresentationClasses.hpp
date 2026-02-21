#pragma once

#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <Eigen/Geometry>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>

// Definimos estructuras de datos auxiliares.

namespace f_vigs_slam
{
    /**
     * @brief Gaussian
     * Representa una gaussiana 3D con sus parámetros principales.
     */
    struct Gaussian
    {
        float3 position;    /// Media
        float3 scale;       /// Varianza (asumimos que no hay correlacion)
        float4 orientation; /// Quaternion (w, x, y, z)
        float3 color;       /// Color RGB [0,1]
        float opacity;      /// Opacidad [0,1]

        // Constructores
        __host__ __device__ Gaussian()
            : position(make_float3(0.f, 0.f, 0.f)),
              scale(make_float3(1.f, 1.f, 1.f)),
              orientation(make_float4(1.f, 0.f, 0.f, 0.f)),
              color(make_float3(1.f, 1.f, 1.f)),
              opacity(1.f)
        {
        }

        __host__ __device__ Gaussian(float3 pos,
                                     float3 scl,
                                     float4 ori,
                                     float3 col,
                                     float op)
            : position(pos),
              scale(scl),
              orientation(ori),
              color(col),
              opacity(op)
        {
        }
    };

    /**
     * @brief Gaussians
     * Contenedor SoA para parámetros de gaussianas con iteradores zip.
     */
    struct Gaussians
    {
        thrust::device_vector<float3> positions;
        thrust::device_vector<float3> scales;
        thrust::device_vector<float4> orientations;
        thrust::device_vector<float3> colors;
        thrust::device_vector<float> opacities;

        void resize(size_t len)
        {
            positions.resize(len);
            scales.resize(len);
            orientations.resize(len);
            colors.resize(len);
            opacities.resize(len);
        }

        typedef thrust::tuple<
            thrust::device_vector<float3>::iterator,
            thrust::device_vector<float3>::iterator,
            thrust::device_vector<float4>::iterator,
            thrust::device_vector<float3>::iterator,
            thrust::device_vector<float>::iterator>
            Tuple;

        typedef thrust::zip_iterator<Tuple> iterator;

        iterator begin()
        {
            return thrust::make_zip_iterator(thrust::make_tuple(
                positions.begin(),
                scales.begin(),
                orientations.begin(),
                colors.begin(),
                opacities.begin()));
        }

        iterator end()
        {
            return thrust::make_zip_iterator(thrust::make_tuple(
                positions.end(),
                scales.end(),
                orientations.end(),
                colors.end(),
                opacities.end()));
        }
    };

    struct IntrinsicParameters
    {
        // f longitud focal, c centro optico
        float2 f, c;
    };

    struct CameraPose {
        // Pose de la camara representada por posicion y orientacion
        // Quaternion formato: (w, x, y, z)
        
        float3 position;     // Posicion (x, y, z)
        float4 orientation;  // Quaternion (w, x, y, z)

        // Constructor por defecto: identidad
        __device__ __host__ CameraPose() 
            : position(make_float3(0.0f, 0.0f, 0.0f)),
              orientation(make_float4(1.0f, 0.0f, 0.0f, 0.0f)) {}

        // Constructor con valores
        __device__ __host__ CameraPose(float3 pos, float4 quat)
            : position(pos), orientation(quat) {}
    };

    /**
     * @brief KeyframeData
     * Almacena un frame con imágenes, pose y parámetros intrínsecos para poder usarse en la optimizacion
     */
    struct KeyframeData
    {
        // ===== IMÁGENES EN GPU =====
        cv::cuda::GpuMat color_img;    ///< Imagen RGB (OpenCV GpuMat - GPU memory)
        cv::cuda::GpuMat depth_img;    ///< Imagen de profundidad (GPU memory)
        
        // ===== POSE Y PARÁMETROS INTRÍNSECOS =====
        CameraPose pose;               ///< Posición + orientación
        IntrinsicParameters intrinsics; ///< focal (fx, fy), centro óptico (cx, cy)
        
        // ===== METADATOS =====
        uint32_t keyframe_id;          ///< ID único del keyframe (0, 1, 2, ...)
        double timestamp;              ///< Tiempo de captura en segundos
        
        // ===== CONSTRUCTORES =====
        /**
         * @brief Constructor vacío - inicializa valores por defecto
         */
        __host__ KeyframeData()
            : color_img(),
              depth_img(),
              pose(),
              intrinsics({make_float2(0.f, 0.f), make_float2(0.f, 0.f)}),
              keyframe_id(0),
              timestamp(0.0)
        {
        }
        
        /**
         * @brief Constructor completo
         * 
         * @param rgb Imagen RGB en GPU
         * @param depth Imagen de profundidad en GPU
         * @param cam_pose Pose de la cámara
         * @param cam_intrinsics Parámetros intrínsecos
         * @param id ID del keyframe
         * @param ts Timestamp de captura
         */
        __host__ KeyframeData(
            const cv::cuda::GpuMat &rgb,
            const cv::cuda::GpuMat &depth,
            const CameraPose &cam_pose,
            const IntrinsicParameters &cam_intrinsics,
            uint32_t id = 0,
            double ts = 0.0)
            : color_img(rgb.clone()),
              depth_img(depth.clone()),
              pose(cam_pose),
              intrinsics(cam_intrinsics),
              keyframe_id(id),
              timestamp(ts)
        {
        }
        
        // ===== GETTERS PARA ACCESO INMUTABLE =====
        /**
         * @brief Obtiene pose (lectura)
         */
        __host__ __device__ inline const CameraPose& getPose() const { return pose; }
        
        /**
         * @brief Obtiene parámetros intrínsecos (lectura)
         */
        __host__ __device__ inline const IntrinsicParameters& getIntrinsics() const { return intrinsics; }
        
        /**
         * @brief Obtiene ID del keyframe
         */
        __host__ __device__ inline uint32_t getId() const { return keyframe_id; }
        
        /**
         * @brief Obtiene timestamp
         */
        __host__ __device__ inline double getTimestamp() const { return timestamp; }
        
        /**
         * @brief Obtiene dimensiones de imagen
         */
        __host__ inline int getWidth() const { return color_img.cols; }
        __host__ inline int getHeight() const { return color_img.rows; }
        
        /**
         * @brief Obtiene tamaño de las imágenes en bytes
         */
        __host__ inline size_t getImageSizeBytes() const 
        { 
            return color_img.step * color_img.rows + depth_img.step * depth_img.rows;
        }
        
        // ===== SETTERS =====
        /**
         * @brief Actualiza la pose
         */
        __host__ inline void setPose(const CameraPose &new_pose) { pose = new_pose; }
        
        /**
         * @brief Actualiza parámetros intrínsecos si es necesario
         */
        __host__ inline void setIntrinsics(const IntrinsicParameters &new_intrinsics) 
        { 
            intrinsics = new_intrinsics; 
        }
    };

    struct ImuData
    {
        // En este struct guardamos la informacion medida y los parametros
        // del modelo dinamico del IMU

        // Mediciones
        Eigen::Vector3d Acc;
        Eigen::Vector3d Gyro;

        // Ruido (noise) y modelo de camino (walk) de acelerometro y giroscopio
        double acc_n;
        double gyr_n;
        double acc_w;
        double gyr_w;
        double g_norm;
    };

    struct PoseOptimizationRgbdData {
        
        // En este struct guardamos la jacobiana*residuo Jtr y la aproximacion
        // del hessiano JtJ para el factor RGB-D

        float Jtr[6];
        float JtJ[21]; // matriz 6x6 simetrica, guardamos solo triangulo sup

        // Definimos el operador suma para GPU y CPU como la suma matricial comun
        // para ambos terminos en la misma operacion
        __device__ __host__ inline PoseOptimizationRgbdData &operator+=(const PoseOptimizationRgbdData &d)
        {
            #pragma unroll
            for(int i=0; i<6; i++)
            {
                Jtr[i]+=d.Jtr[i];
            }
            #pragma unroll
            for(int i=0; i<21; i++)
            {
                JtJ[i]+=d.JtJ[i];
            }

            return *this;
        }
    };
    
    struct PoseOptimizationMetrics {
        // En este struct guardamos metricas de optimizacion de pose

        // NO IMPLEMENTADA AUN
    };

}