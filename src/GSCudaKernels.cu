#include <f_vigs_slam/GSCudaKernels.cuh>
#include <math.h>
#include <algorithm>
#include <cub/cub.cuh>  // para reduce

namespace f_vigs_slam
{
    // Implementamos los kernels CUDA para operaciones paralelizables
    // El archivo esta ordenado del siguiente modo:
    // - Forward pass kernels (renderizacion): todo lo necesario para renderizar las gaussianas
    // - Backward pass kernels (optimizacion): todo lo necesario para computar gradientes
    // - Funciones auxiliares

    // ============================================================================
    // Inicializacion de Gaussianas desde RGB-D
    // ============================================================================
    __device__ inline float3 rotateByQuaternion(const float4 &q, const float3 &v)
    {
        // q = (w, x, y, z)
        float3 qv = make_float3(q.y, q.z, q.w);
        float3 t = 2.0f * cross(qv, v);
        return v + q.x * t + cross(qv, t);
    }

    __device__ inline float3 normalizeVec3(const float3 &v)
    {
        float n = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
        if (n > 1e-8f)
        {
            float inv = 1.0f / n;
            return make_float3(v.x * inv, v.y * inv, v.z * inv);
        }
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    __device__ inline float4 quatMultiply(const float4 &a, const float4 &b)
    {
        // Quaternion formato (w, x, y, z)
        float aw = a.x, ax = a.y, ay = a.z, az = a.w;
        float bw = b.x, bx = b.y, by = b.z, bz = b.w;
        float4 out;
        out.x = aw * bw - ax * bx - ay * by - az * bz;
        out.y = aw * bx + ax * bw + ay * bz - az * by;
        out.z = aw * by - ax * bz + ay * bw + az * bx;
        out.w = aw * bz + ax * by - ay * bx + az * bw;
        return out;
    }

    __device__ inline float4 quatFromTwoVectors(const float3 &from, const float3 &to)
    {
        float3 u = normalizeVec3(from);
        float3 v = normalizeVec3(to);
        float dot = u.x * v.x + u.y * v.y + u.z * v.z;

        if (dot < -0.999999f)
        {
            float3 axis = (fabsf(u.x) < 0.1f) ? make_float3(1.0f, 0.0f, 0.0f)
                                              : make_float3(0.0f, 1.0f, 0.0f);
            float3 ortho = normalizeVec3(cross(u, axis));
            return make_float4(0.0f, ortho.x, ortho.y, ortho.z);
        }

        float3 c = cross(u, v);
        float4 q = make_float4(1.0f + dot, c.x, c.y, c.z);
        float n = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        if (n > 1e-8f)
        {
            float inv = 1.0f / n;
            q.x *= inv;
            q.y *= inv;
            q.z *= inv;
            q.w *= inv;
        }
        else
        {
            q = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
        }
        return q;
    }

    __global__ void initGaussiansFromRgbd_kernel(
        float3 *positions,
        float3 *scales,
        float4 *orientations,
        float3 *colors,
        float *opacities,
        uint32_t *instanceCounter,
        uint32_t maxGaussians,
        const uchar3 *rgb,
        size_t rgb_step,
        const float *depth,
        size_t depth_step,
        int width,
        int height,
        IntrinsicParameters intrinsics,
        CameraPose cameraPose,
        uint32_t sample_dx,
        uint32_t sample_dy,
        float init_scale,
        float init_opacity)
    {
        // Calculamos el tamaño de la grilla de muestreo
        int sample_w = (width + sample_dx - 1) / sample_dx;
        int sample_h = (height + sample_dy - 1) / sample_dy;

        // Calculamos la celda en la grilla
        int sx = blockIdx.x * blockDim.x + threadIdx.x;
        int sy = blockIdx.y * blockDim.y + threadIdx.y;

        if (sx >= sample_w || sy >= sample_h) return;

        // Calculamos la posicion del pixel
        int u = sx * sample_dx;
        int v = sy * sample_dy;
        if (u >= width || v >= height) return;

        // Tomamos los valores de RGB-D
        const unsigned char *rgb_row = reinterpret_cast<const unsigned char *>(rgb) + v * rgb_step;
        const uchar3 *rgb_row3 = reinterpret_cast<const uchar3 *>(rgb_row);
        uchar3 bgr = rgb_row3[u];

        const unsigned char *depth_row = reinterpret_cast<const unsigned char *>(depth) + v * depth_step;
        const float *depth_row_f = reinterpret_cast<const float *>(depth_row);
        float z = depth_row_f[u];
        if (!(z > 0.0f) || isinf(z) || isnan(z)) return;

        // Transformamos a coordenadas 3D
        float x = (static_cast<float>(u) - intrinsics.c.x) * z / intrinsics.f.x;
        float y = (static_cast<float>(v) - intrinsics.c.y) * z / intrinsics.f.y;
        float3 pos_cam = make_float3(x, y, z);

        float3 pos_world = cameraPose.position + rotateByQuaternion(cameraPose.orientation, pos_cam);

        // Añadimos al contador de gaussianas
        uint32_t idx = atomicAdd(instanceCounter, 1u);
        if (idx >= maxGaussians) return;

        // Inicializamos la gaussiana
        positions[idx] = pos_world;
        scales[idx] = make_float3(init_scale, init_scale, init_scale);
        orientations[idx] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
        colors[idx] = make_float3(bgr.z / 255.0f, bgr.y / 255.0f, bgr.x / 255.0f);
        opacities[idx] = init_opacity;
    }



    // ============================================================================
    // Forward pass kernels (Renderizacion)
    // ============================================================================
    
    __device__ inline void quaternionToMatrix(const float4 &q, float R[3][3])
    {
        float w = q.w, x = q.x, y = q.y, z = q.z;
        
        R[0][0] = 1.0f - 2.0f * (y * y + z * z);
        R[0][1] = 2.0f * (x * y - w * z);
        R[0][2] = 2.0f * (x * z + w * y);
        
        R[1][0] = 2.0f * (x * y + w * z);
        R[1][1] = 1.0f - 2.0f * (x * x + z * z);
        R[1][2] = 2.0f * (y * z - w * x);
        
        R[2][0] = 2.0f * (x * z - w * y);
        R[2][1] = 2.0f * (y * z + w * x);
        R[2][2] = 1.0f - 2.0f * (x * x + y * y);
    }
    
    __device__ inline float3 matrixVectorMul(float R[3][3], float3 v)
    {
        return make_float3(
            R[0][0] * v.x + R[0][1] * v.y + R[0][2] * v.z,
            R[1][0] * v.x + R[1][1] * v.y + R[1][2] * v.z,
            R[2][0] * v.x + R[2][1] * v.y + R[2][2] * v.z
        );
    }

    __global__ void projectGaussiansWorldToScreen_kernel(
        float2 *positions_2d,
        float3 *covariances_2d,
        float3 *inv_covariances_2d,
        float *depths,
        float2 *p_hats,
        const float3 *positions_world,
        const float3 *scales,
        const float4 *orientations,
        CameraPose camera_pose,
        IntrinsicParameters intrinsics,
        int width,
        int height,
        int n_gaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_gaussians) return;
        
        // ============================================================
        // Paso 1: Transformar de mundo a cámara
        // ============================================================
        float3 pos_world = positions_world[idx];
        float3 pos_diff = pos_world - camera_pose.position;
        
        // Rotar: pos_cam = R_cam^T * (pos_world - t_cam)
        float R_cam[3][3];
        quaternionToMatrix(camera_pose.orientation, R_cam);
        
        // Invertir rotación para pasar a frame de cámara
        // R_cam^T = transpuesta
        float3 pos_cam;
        pos_cam.x = R_cam[0][0] * pos_diff.x + R_cam[1][0] * pos_diff.y + R_cam[2][0] * pos_diff.z;
        pos_cam.y = R_cam[0][1] * pos_diff.x + R_cam[1][1] * pos_diff.y + R_cam[2][1] * pos_diff.z;
        pos_cam.z = R_cam[0][2] * pos_diff.x + R_cam[1][2] * pos_diff.y + R_cam[2][2] * pos_diff.z;
        
        // Filtrar: solo procesamos gaussianas enfrente de la cámara
        if (pos_cam.z <= 0.0f) return;
        
        // ============================================================
        // Paso 2: Proyectar a pantalla
        // ============================================================
        float inv_z = 1.0f / pos_cam.z;
        float u = intrinsics.f.x * (pos_cam.x * inv_z) + intrinsics.c.x;
        float v = intrinsics.f.y * (pos_cam.y * inv_z) + intrinsics.c.y;
        
        // Filtrar: solo procesamos si está en viewport
        if (u < 0.0f || u >= width || v < 0.0f || v >= height) return;
        
        positions_2d[idx] = make_float2(u, v);
        depths[idx] = pos_cam.z;
        
        // ============================================================
        // Paso 3: Calcular covarianza 3D en frame de cámara
        // ============================================================
        // Σ_cam = R_cam^T * (R_gauss * S * S^T * R_gauss^T) * R_cam
        float3 scale = scales[idx];
        float4 orientation = orientations[idx];
        
        float R_gauss[3][3];
        quaternionToMatrix(orientation, R_gauss);
        
        // M = R_gauss * diag(scale)
        float M[3][3];
        for (int i = 0; i < 3; i++) {
            M[i][0] = R_gauss[i][0] * scale.x;
            M[i][1] = R_gauss[i][1] * scale.y;
            M[i][2] = R_gauss[i][2] * scale.z;
        }
        
        // Σ = M * M^T
        float Sigma[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = i; j < 3; j++) {
                Sigma[i][j] = M[i][0] * M[j][0] + M[i][1] * M[j][1] + M[i][2] * M[j][2];
                if (i != j) Sigma[j][i] = Sigma[i][j];
            }
        }
        
        // Σ_cam = R_cam^T * Σ * R_cam (transpuesta aplicada a rotación)
        float Sigma_cam[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Sigma_cam[i][j] = 0.0f;
                for (int k = 0; k < 3; k++) {
                    // R_cam^T[i][k] = R_cam[k][i]
                    for (int l = 0; l < 3; l++) {
                        Sigma_cam[i][j] += R_cam[k][i] * Sigma[k][l] * R_cam[l][j];
                    }
                }
            }
        }
        
        // ============================================================
        // Paso 4: Proyectar a 2D con Jacobiano
        // ============================================================
        // J = [fx/z    0   -fx*x/z²]
        //     [  0   fy/z  -fy*y/z²]
        float J[2][3];
        J[0][0] = intrinsics.f.x * inv_z;
        J[0][1] = 0.0f;
        J[0][2] = -intrinsics.f.x * pos_cam.x * inv_z * inv_z;
        
        J[1][0] = 0.0f;
        J[1][1] = intrinsics.f.y * inv_z;
        J[1][2] = -intrinsics.f.y * pos_cam.y * inv_z * inv_z;
        
        // Σ_2d = J * Σ_cam[0:2, 0:2] * J^T
        float Sigma_2d[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = i; j < 2; j++) {
                Sigma_2d[i][j] = 0.0f;
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        Sigma_2d[i][j] += J[i][k] * Sigma_cam[k][l] * J[j][l];
                    }
                }
                if (i != j) Sigma_2d[j][i] = Sigma_2d[i][j];
            }
        }
        
        covariances_2d[idx] = make_float3(Sigma_2d[0][0], Sigma_2d[1][1], Sigma_2d[0][1]);

        float inv_cov_xx, inv_cov_yy, inv_cov_xy;
        invert2x2(Sigma_2d[0][0], Sigma_2d[1][1], Sigma_2d[0][1], inv_cov_xx, inv_cov_yy, inv_cov_xy);
        inv_covariances_2d[idx] = make_float3(inv_cov_xx, inv_cov_yy, inv_cov_xy);
        
        // ============================================================
        // Paso 5: Calcular p_hat para depth ordering
        // ============================================================
        // p_hat refleja cómo cambia la profundidad con desplazamiento en pixel
        p_hats[idx] = make_float2(pos_cam.x * inv_z * inv_z, pos_cam.y * inv_z * inv_z);
    }

    __global__ void projectGaussiansKernel(
        float2 *positions_2d,
        float3 *covariances_2d,
        float *depths,
        float2 *pHats,
        const float3 *positions_cam,
        const float3 *covariances_cam,
        const float *intrinsics,
        int width,
        int height,
        int n_gaussians)
    {
        // Un indice por cada gaussiana
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_gaussians) return;

        // Obtenemos la posicion 3D y covarianza 3D de la gaussiana actual
        float3 pos_cam = positions_cam[idx];
        float3 cov_cam = covariances_cam[idx];

        // Proyectamos a 2D usando intrinsecos de la camara
        float fx = intrinsics[0];
        float fy = intrinsics[1];
        float cx = intrinsics[2];
        float cy = intrinsics[3];

        float z = pos_cam.z;
        float x = pos_cam.x;
        float y = pos_cam.y;

        float sobre_z = 1.0f / z;
        float u = fx * (x * sobre_z) + cx;
        float v = fy * (y * sobre_z) + cy;

        positions_2d[idx] = make_float2(u, v);
        depths[idx] = z;

        // Transformamos covarianza 3D a 2D usando el Jacobiano de proyeccion
        float var_xx_2d = (fx * fx * cov_cam.x) * (sobre_z * sobre_z);
        float var_yy_2d = (fy * fy * cov_cam.y) * (sobre_z * sobre_z);
        float var_xy_2d = 0.0f; // Asumimos independencia para simplificar

        covariances_2d[idx] = make_float3(var_xx_2d, var_yy_2d, var_xy_2d);

        // Calcular pHat para orden de profundidad por pixel
        // Siguiendo formulacion de radegs: d = z_c + pHat · (u_c - u, v_c - v)
        // En espacio de rayos, v' = (0,0,1), y queremos:
        // pHat = (z_c / t_c) * (v'^T * Sigma'^{-1}) / (v'^T * Sigma'^{-1} * v')
        // Para covarianza 2D, la tercera fila/columna corresponde a la direccion z
        // Aproximacion: pHat refleja como cambia la profundidad con desplazamiento en imagen
        float t_c = sqrtf(x * x + y * y + z * z);
        float inv_t_c = 1.0f / t_c;
        
        // Inversion de covarianza 2D
        float inv_cov_xx, inv_cov_yy, inv_cov_xy;
        invert2x2(var_xx_2d, var_yy_2d, var_xy_2d, inv_cov_xx, inv_cov_yy, inv_cov_xy);
        
        // v' en espacio de rayos es (0, 0, 1), por lo que v'^T * Sigma'^{-1} * v' = inv_cov_zz
        // Para 2D, usamos la aproximacion: derivada de profundidad respecto a (u,v)
        // Componente en u: como z varia con cambios en u
        // Componente en v: como z varia con cambios en v
        float denominator = inv_cov_xx + inv_cov_yy + 2.0f * fabsf(inv_cov_xy) + 1e-6f;
        float pHat_x = (z * inv_t_c) * (inv_cov_xx / denominator);
        float pHat_y = (z * inv_t_c) * (inv_cov_yy / denominator);
        
        pHats[idx] = make_float2(pHat_x, pHat_y);
    }


    // Kernel para contar cuántos tiles cubre cada gaussiana
    // Pre-contamos para saber de antemano la cantidad de memoria necesaria para los hashes
    __global__ void countTilesPerGaussian_kernel(
        uint32_t *tile_counts,          // Salida: número de tiles por gaussiana
        const float2 *positions_2d,     // Entrada: posiciones proyectadas (centro)
        const float3 *covariances_2d,   // Entrada: covarianzas 2D (xx, yy, xy)
        int width,
        int height,
        int n_gaussians,
        int num_tiles_x,
        int num_tiles_y,
        float radius_sigma)             // Radio en múltiplos de sigma (ej: 3.0f)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_gaussians) return;

        float2 pos = positions_2d[idx];
        float3 cov = covariances_2d[idx];

        // Calcular el radio de cobertura basado en covarianza (3-sigma)
        float sigma_x = sqrtf(cov.x);
        float sigma_y = sqrtf(cov.y);
        float radius_x = radius_sigma * sigma_x;
        float radius_y = radius_sigma * sigma_y;

        // Calcular bounding box de la gaussiana
        float bb_min_x = max(0.0f, pos.x - radius_x);
        float bb_max_x = min((float)(width - 1), pos.x + radius_x);
        float bb_min_y = max(0.0f, pos.y - radius_y);
        float bb_max_y = min((float)(height - 1), pos.y + radius_y);

        // Convertir a tiles
        int tile_min_x = (int)(bb_min_x / TILE_SIZE);
        int tile_max_x = (int)(bb_max_x / TILE_SIZE);
        int tile_min_y = (int)(bb_min_y / TILE_SIZE);
        int tile_max_y = (int)(bb_max_y / TILE_SIZE);

        // Garantizar rangos válidos
        tile_min_x = max(0, min(tile_min_x, num_tiles_x - 1));
        tile_max_x = max(0, min(tile_max_x, num_tiles_x - 1));
        tile_min_y = max(0, min(tile_min_y, num_tiles_y - 1));
        tile_max_y = max(0, min(tile_max_y, num_tiles_y - 1));

        // Contar tiles que cubre
        uint32_t count = (tile_max_x - tile_min_x + 1) * (tile_max_y - tile_min_y + 1);
        tile_counts[idx] = count;
    }

    // Kernel para generar hashes por cada tile que cubre la gaussiana
    __global__ void generateTileHashes_kernel(
        uint64_t *hashes,               // Salida: hashes para todos los gaussianas×tiles
        uint32_t *gaussian_indices,     // Salida: índice de gaussiana para cada hash
        const uint32_t *tile_offsets,   // Entrada: offsets acumulativos de tile_counts
        const float2 *positions_2d,     // Entrada: posiciones proyectadas (centro)
        const float3 *covariances_2d,   // Entrada: covarianzas 2D (xx, yy, xy)
        const float *depths,            // Entrada: profundidades Z
        const float2 *pHats,            // Entrada: coeficientes para profundidad por pixel
        int width,
        int height,
        int n_gaussians,
        int num_tiles_x,
        int num_tiles_y,
        float radius_sigma)             // Radio en múltiplos de sigma (ej: 3.0f)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_gaussians) return;

        float2 pos = positions_2d[idx];
        float3 cov = covariances_2d[idx];
        float z = depths[idx];
        float2 pHat = pHats[idx];

        // Calcular el radio de cobertura
        float sigma_x = sqrtf(cov.x);
        float sigma_y = sqrtf(cov.y);
        float radius_x = radius_sigma * sigma_x;
        float radius_y = radius_sigma * sigma_y;

        // Calcular bounding box
        float bb_min_x = max(0.0f, pos.x - radius_x);
        float bb_max_x = min((float)(width - 1), pos.x + radius_x);
        float bb_min_y = max(0.0f, pos.y - radius_y);
        float bb_max_y = min((float)(height - 1), pos.y + radius_y);

        // Convertir a tiles
        int tile_min_x = (int)(bb_min_x / TILE_SIZE);
        int tile_max_x = (int)(bb_max_x / TILE_SIZE);
        int tile_min_y = (int)(bb_min_y / TILE_SIZE);
        int tile_max_y = (int)(bb_max_y / TILE_SIZE);

        tile_min_x = max(0, min(tile_min_x, num_tiles_x - 1));
        tile_max_x = max(0, min(tile_max_x, num_tiles_x - 1));
        tile_min_y = max(0, min(tile_min_y, num_tiles_y - 1));
        tile_max_y = max(0, min(tile_max_y, num_tiles_y - 1));

        // Generar un hash por cada tile que cubre esta gaussiana
        uint32_t write_offset = tile_offsets[idx];

        int hash_idx = 0;
        for (int ty = tile_min_y; ty <= tile_max_y; ty++)
        {
            for (int tx = tile_min_x; tx <= tile_max_x; tx++)
            {
                // Calcular profundidad en el centro del tile para ordenamiento
                float tile_center_x = (tx + 0.5f) * TILE_SIZE;
                float tile_center_y = (ty + 0.5f) * TILE_SIZE;
                float dx = tile_center_x - pos.x;
                float dy = tile_center_y - pos.y;
                float depth_at_tile = z + pHat.x * dx + pHat.y * dy;
                
                uint32_t tile_id = ty * num_tiles_x + tx;
                uint64_t depth_bits = __float_as_uint(depth_at_tile);
                uint64_t hash = ((uint64_t)tile_id << 32) | depth_bits;

                hashes[write_offset + hash_idx] = hash;
                gaussian_indices[write_offset + hash_idx] = idx;

                hash_idx++;
            }
        }
    }

    __global__ void computeIndicesRanges_kernel(
        uint2 *ranges,
        const uint64_t *hashes,
        uint32_t n_instances,
        int num_tiles)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_instances) return;

        // Extraemos tile_id del hash (bits altos)
        // Se asume que los hashes están ORDENADOS por thrust::sort_by_key
        // n_instances = total_hashes (puede ser mayor a #gaussianas porque a priori cada una
        // puede cubrir varios tiles segun su radio)
        uint32_t tile_id = (uint32_t)(hashes[idx] >> 32);

        // Primer elemento
        if (idx == 0)
        {
            ranges[tile_id].x = 0;
            return;
        }

        // Comprobar si el tile_id cambió respecto al anterior
        uint32_t prev_tile_id = (uint32_t)(hashes[idx - 1] >> 32);

        if (prev_tile_id != tile_id)
        {
            // Tile anterior termina aquí
            ranges[prev_tile_id].y = idx;
            // Nuevo tile empieza aquí
            ranges[tile_id].x = idx;
        }

        // Último elemento: cierra el último tile
        if (idx == n_instances - 1)
        {
            ranges[tile_id].y = n_instances;
        }
    }

    // REPASAR: invertir covarianzas de antemano y cargar por shared
    __global__ void forwardPassTileKernel(
        float3 *output_color,
        float *output_depth,
        const uint32_t *tile_gaussian_indices,
        const uint2 *tile_ranges,
        const float2 *positions_2d,
        const float3 *covariances_2d,
        const float3 *colors,
        const float *alphas,
        const float *depths,
        const float2 *pHats,
        int width,
        int height,
        int num_tiles_x,
        int num_tiles_y
        )
    {
        // Como tenemos que iterar sobre las gaussianas de cada tile, que no son necesariamente la misma
        // cantidad ni tenemos cota superior, usamos shared memory y resolvemos de a batches

        __shared__ float2 s_positions_2d[TILE_SIZE * TILE_SIZE];
        __shared__ float3 s_covariances_2d[TILE_SIZE * TILE_SIZE];
        __shared__ float3 s_colors[TILE_SIZE * TILE_SIZE];
        __shared__ float s_alphas[TILE_SIZE * TILE_SIZE];
        __shared__ float s_depths[TILE_SIZE * TILE_SIZE];
        __shared__ float2 s_pHats[TILE_SIZE * TILE_SIZE];

        // Un bloque por cada tile
        // Un thread por cada pixel dentro del tile

        int pixel_x = blockIdx.x * TILE_SIZE + threadIdx.x;
        int pixel_y = blockIdx.y * TILE_SIZE + threadIdx.y;
        bool inside = (pixel_x < width && pixel_y < height);

        int pixel_idx = pixel_y * width + pixel_x;
        float px = (float)pixel_x;
        float py = (float)pixel_y;

        float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);
        float pixel_alpha = 0.0f;
        float pixel_depth = 1e10f;

        // Iteramos solo sobre las gaussianas relevantes para este tile
        bool has_median_depth = false;
        bool done = !inside;

        int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
        int range_start = tile_ranges[tid].x;
        int range_end   = tile_ranges[tid].y;
        int n_gaussians_tile = range_end - range_start;
        int block_size = TILE_SIZE * TILE_SIZE;

        for (int base = 0; base < n_gaussians_tile; base += block_size)
        {
            int local_idx = base + tid;
            if (local_idx < n_gaussians_tile)
            {
                int g = tile_gaussian_indices[range_start + local_idx];
                s_positions_2d[tid] = positions_2d[g];
                s_covariances_2d[tid] = covariances_2d[g];
                s_colors[tid] = colors[g];
                s_alphas[tid] = alphas[g];
                s_depths[tid] = depths[g];
                s_pHats[tid] = pHats[g];
            }
            __syncthreads();

            int batch_count = min(block_size, n_gaussians_tile - base);
            for (int i = 0; i < batch_count; i++)
            {
                if (!done)
                {
                    float2 pos_2d = s_positions_2d[i];
                    float3 cov_2d = s_covariances_2d[i];
                    float3 color = s_colors[i];
                    float alpha = s_alphas[i];
                    float z_c = s_depths[i];
                    float2 pHat = s_pHats[i];

                    float dx = px - pos_2d.x;
                    float dy = py - pos_2d.y;
                    

                    float inv_cov_xx, inv_cov_yy, inv_cov_xy;
                    invert2x2(cov_2d.x, cov_2d.y, cov_2d.z, inv_cov_xx, inv_cov_yy, inv_cov_xy);

                    float gauss_val = evalGaussian2D(dx, dy, inv_cov_xx, inv_cov_yy, inv_cov_xy);
                    float weighted_alpha = alpha * gauss_val;

                    // Transmitancia previa a acumular la gaussiana g
                    float T_before = 1.0f - pixel_alpha;

                    pixel_color.x += (1.0f - pixel_alpha) * weighted_alpha * color.x;
                    pixel_color.y += (1.0f - pixel_alpha) * weighted_alpha * color.y;
                    pixel_color.z += (1.0f - pixel_alpha) * weighted_alpha * color.z;

                    pixel_alpha += (1.0f - pixel_alpha) * weighted_alpha;

                    // Transmitancia luego de acumularla
                    float T_after = 1.0f - pixel_alpha;

                    // Profundidad mediana: la definimos como la que provoca que se cruce el umbral 0.5
                    // Ahora usa la profundidad calculada por pixel con pHat usando la derivacion
                    // explicada a partir del espacio de rayos
                    if (!has_median_depth && T_before > 0.5f && T_after <= 0.5f)
                    {
                        pixel_depth = z_c + pHat.x * dx + pHat.y * dy;
                        has_median_depth = true;
                    }

                    if (pixel_alpha >= 0.99f)
                    {
                        done = true;
                    }
                }
            }

            __syncthreads();
        }

        if (inside)
        {
            output_color[pixel_idx] = pixel_color;
            output_depth[pixel_idx] = pixel_depth;
        }

    }

    /*  __global__ void forwardPassKernel(
        float3 *output_color,           // Salida: imagen RGB renderizada
        float *output_depth,            // Salida: mapa de profundidad
        const float2 *positions_2d,     // Entrada: posiciones proyectadas [n_gaussians]
        const float3 *covariances_2d,   // Entrada: covarianzas 2D [n_gaussians]
        const float3 *colors,           // Entrada: color RGB de cada Gaussian [n_gaussians]
        const float *alphas,            // Entrada: opacidad de cada Gaussian [n_gaussians]
        int width,
        int height,
        int n_gaussians)
    {
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

        if (idx_x >= width || idx_y >= height) return;

        int pixel_idx = idx_y * width + idx_x;
        float px = (float)idx_x;
        float py = (float)idx_y;

        float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);
        float pixel_alpha = 0.0f;
        float pixel_depth = 1e10f;



        for (int g = 0; g < n_gaussians; g++)
        {
            float2 pos_2d = positions_2d[g];
            float3 cov_2d = covariances_2d[g];
            float3 color = colors[g];
            float alpha = alphas[g];

            float dx = px - pos_2d.x;
            float dy = py - pos_2d.y;

            float inv_cov_xx, inv_cov_yy, inv_cov_xy;
            invert2x2(cov_2d.x, cov_2d.y, cov_2d.z, inv_cov_xx, inv_cov_yy, inv_cov_xy);

            float gauss_val = evalGaussian2D(dx, dy, inv_cov_xx, inv_cov_yy, inv_cov_xy);

            float weighted_alpha = alpha * gauss_val;

            pixel_color.x += (1.0f - pixel_alpha) * weighted_alpha * color.x;
            pixel_color.y += (1.0f - pixel_alpha) * weighted_alpha * color.y;
            pixel_color.z += (1.0f - pixel_alpha) * weighted_alpha * color.z;

            pixel_alpha += (1.0f - pixel_alpha) * weighted_alpha;

            if (pixel_alpha >= 0.99f) break; // En muchos casos la contribucion se vuelve casi nula en cierto punto
        }

        output_color[pixel_idx] = pixel_color;
        output_depth[pixel_idx] = pixel_depth;
    } */


    // ============================================================================
    // Backward pass kernels (Optimizacion)
    // ============================================================================

    __global__ void getRgbdPoseJacobians(
        PoseOptimizationRgbdData *output_posedata,
        const uint2 *ranges,
        const uint32_t *indices,
        const float2 *positions_2d,
        const float *depths,
        const float3 *inv_covariances_2d,
        const float2 *p_hats,
        const float3 *colors,
        const float *alphas,
        const float3 *grad_x,
        const float3 *grad_y,
        const float3 *observed_rgb,
        const float *observed_depth,
        CameraPose camera_pose,
        IntrinsicParameters intrinsics,
        float3 bg_color,
        float alpha_thresh,
        float color_thresh,
        float depth_thresh,
        int width, int height,
        int num_tiles_x, int num_tiles_y)
    {
        // Un thread por pixel, un bloque por tile
        int x = blockIdx.x * TILE_SIZE + threadIdx.x;
        int y = blockIdx.y * TILE_SIZE + threadIdx.y;
        int tile_idx = blockIdx.y * num_tiles_x + blockIdx.x;
        int pixel_idx = y * width + x;
        int tid = threadIdx.y * TILE_SIZE + threadIdx.x;

        float img_depth = 0.0f;
        bool in_bounds = (x < width && y < height);
        if (in_bounds) {
            img_depth = observed_depth[pixel_idx];
        }
        bool inside = in_bounds && img_depth > 0.1f;

        float local_JtJ[21] = {0.0f};
        float local_Jtr[6] = {0.0f};

        __shared__ float2 s_positions[TILE_SIZE * TILE_SIZE];
        __shared__ float3 s_invSigmas[TILE_SIZE * TILE_SIZE];
        __shared__ float3 s_colors[TILE_SIZE * TILE_SIZE];
        __shared__ float s_alphas[TILE_SIZE * TILE_SIZE];
        __shared__ float s_depths[TILE_SIZE * TILE_SIZE];
        __shared__ float2 s_pHats[TILE_SIZE * TILE_SIZE];

        uint2 range = ranges[tile_idx];
        int block_size = TILE_SIZE * TILE_SIZE;
        int n_total = (int)(range.y - range.x);
        const float inv_255 = 1.0f / 255.0f;

        float3 color = make_float3(0.0f, 0.0f, 0.0f);
        float depth = 0.0f;
        float final_T = 1.0f;
        uint32_t n_contrib = 0;

        if (n_total > 0) {
            float T = 1.0f;
            uint32_t contributor = 0;
            uint32_t last_contributor = 0;
            bool done = !inside;

            for (int base = 0; base < n_total; base += block_size) {
                int local_idx = base + tid;
                if (local_idx < n_total) {
                    uint32_t gid = indices[range.x + local_idx];
                    s_positions[tid] = positions_2d[gid];
                    s_invSigmas[tid] = inv_covariances_2d[gid];
                    s_colors[tid] = colors[gid];
                    s_alphas[tid] = alphas[gid];
                    s_depths[tid] = depths[gid];
                    s_pHats[tid] = p_hats[gid];
                }
                __syncthreads();

                int batch_count = min(block_size, n_total - base);
                for (int i = 0; i < batch_count; ++i) {
                    if (done || (i + base) >= n_total) {
                        continue;
                    }
                    contributor++;
                    float dx = s_positions[i].x - (float)x;
                    float dy = s_positions[i].y - (float)y;
                    float3 invSigma = s_invSigmas[i];
                    float v = invSigma.x * dx * dx + 2.0f * invSigma.y * dx * dy + invSigma.z * dy * dy;
                    float alpha_i = fminf(0.99f, s_alphas[i] * expf(-0.5f * v));
                    if (alpha_i < inv_255 || v <= 0.0f) {
                        continue;
                    }

                    float test_T = T * (1.0f - alpha_i);
                    if (test_T < 0.0001f) {
                        done = true;
                        continue;
                    }

                    color.x += s_colors[i].x * alpha_i * T;
                    color.y += s_colors[i].y * alpha_i * T;
                    color.z += s_colors[i].z * alpha_i * T;

                    float d = s_depths[i] + dx * s_pHats[i].x + dy * s_pHats[i].y;
                    if (T > 0.5f && test_T < 0.5f) {
                        depth = d;
                    }

                    T = test_T;
                    last_contributor = contributor;
                }
                __syncthreads();
            }

            if (inside) {
                final_T = T;
                n_contrib = last_contributor;
                color.x += T * bg_color.x;
                color.y += T * bg_color.y;
                color.z += T * bg_color.z;
            }
        }

        inside = inside && (final_T < alpha_thresh);

        float3 color_error = make_float3(0.0f, 0.0f, 0.0f);
        float depth_error = 0.0f;
        float wc = 0.0f;

        float R[3][3];

        if (inside) {
            quaternionToMatrix(camera_pose.orientation, R);
            float3 observed_c = observed_rgb[pixel_idx];
            color_error = make_float3(color.x - observed_c.x,
                                      color.y - observed_c.y,
                                      color.z - observed_c.z);
            depth_error = (img_depth > 0.1f) ? (depth - img_depth) : 0.0f;

            float color_loss = sqrtf(color_error.x * color_error.x +
                                     color_error.y * color_error.y +
                                     color_error.z * color_error.z);
            wc = (color_loss < color_thresh) ? 1.0f : color_thresh / color_loss;

            float3 ray = make_float3((x - intrinsics.c.x) / intrinsics.f.x,
                                     (y - intrinsics.c.y) / intrinsics.f.y,
                                     1.0f);

            float ray_cross[9];
            ray_cross[0] = 0.0f;      ray_cross[1] = -ray.z;  ray_cross[2] = ray.y;
            ray_cross[3] = ray.z;     ray_cross[4] = 0.0f;    ray_cross[5] = -ray.x;
            ray_cross[6] = -ray.y;    ray_cross[7] = ray.x;   ray_cross[8] = 0.0f;

            float3 gradX = grad_x[pixel_idx];
            float3 gradY = grad_y[pixel_idx];

            float Jt_pix[6];
            float inv_img_depth = 1.0f / img_depth;
            Jt_pix[0] = intrinsics.f.x * inv_img_depth;
            Jt_pix[1] = 0.0f;
            Jt_pix[2] = 0.0f;
            Jt_pix[3] = intrinsics.f.y * inv_img_depth;
            Jt_pix[4] = -intrinsics.f.x * ray.x * inv_img_depth;
            Jt_pix[5] = -intrinsics.f.y * ray.y * inv_img_depth;

            float Jt_cam_pix[9];
            Jt_cam_pix[0] = Jt_pix[0] * gradX.x + Jt_pix[1] * gradY.x;
            Jt_cam_pix[1] = Jt_pix[0] * gradX.y + Jt_pix[1] * gradY.y;
            Jt_cam_pix[2] = Jt_pix[0] * gradX.z + Jt_pix[1] * gradY.z;
            Jt_cam_pix[3] = Jt_pix[2] * gradX.x + Jt_pix[3] * gradY.x;
            Jt_cam_pix[4] = Jt_pix[2] * gradX.y + Jt_pix[3] * gradY.y;
            Jt_cam_pix[5] = Jt_pix[2] * gradX.z + Jt_pix[3] * gradY.z;
            Jt_cam_pix[6] = Jt_pix[4] * gradX.x + Jt_pix[5] * gradY.x;
            Jt_cam_pix[7] = Jt_pix[4] * gradX.y + Jt_pix[5] * gradY.y;
            Jt_cam_pix[8] = Jt_pix[4] * gradX.z + Jt_pix[5] * gradY.z;

            float JtJ_cam_pix[9];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; ++k) {
                        sum += Jt_cam_pix[r * 3 + k] * Jt_cam_pix[c * 3 + k];
                    }
                    JtJ_cam_pix[r * 3 + c] = wc * sum;
                }
            }

            float Jtr_cam_pix[3];
            Jtr_cam_pix[0] = -wc * (Jt_cam_pix[0] * color_error.x + Jt_cam_pix[1] * color_error.y + Jt_cam_pix[2] * color_error.z);
            Jtr_cam_pix[1] = -wc * (Jt_cam_pix[3] * color_error.x + Jt_cam_pix[4] * color_error.y + Jt_cam_pix[5] * color_error.z);
            Jtr_cam_pix[2] = -wc * (Jt_cam_pix[6] * color_error.x + Jt_cam_pix[7] * color_error.y + Jt_cam_pix[8] * color_error.z);

            float ld = fabsf(depth_error);
            float wd = (ld < depth_thresh) ? 1.0f : depth_thresh / ld;
            wd /= img_depth;
            JtJ_cam_pix[0] += wd * ray.x * ray.x;
            JtJ_cam_pix[1] += wd * ray.x * ray.y;
            JtJ_cam_pix[2] += wd * ray.x * ray.z;
            JtJ_cam_pix[3] += wd * ray.y * ray.x;
            JtJ_cam_pix[4] += wd * ray.y * ray.y;
            JtJ_cam_pix[5] += wd * ray.y * ray.z;
            JtJ_cam_pix[6] += wd * ray.z * ray.x;
            JtJ_cam_pix[7] += wd * ray.z * ray.y;
            JtJ_cam_pix[8] += wd * ray.z * ray.z;

            Jtr_cam_pix[0] += wd * depth_error * ray.x;
            Jtr_cam_pix[1] += wd * depth_error * ray.y;
            Jtr_cam_pix[2] += wd * depth_error * ray.z;

            float Jpose_pix[18];
            Jpose_pix[0] = R[0][0];  Jpose_pix[1] = R[0][1];  Jpose_pix[2] = R[0][2];
            Jpose_pix[3] = R[1][0];  Jpose_pix[4] = R[1][1];  Jpose_pix[5] = R[1][2];
            Jpose_pix[6] = R[2][0];  Jpose_pix[7] = R[2][1];  Jpose_pix[8] = R[2][2];

            float z_ray_cross_pix[9];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    z_ray_cross_pix[r * 3 + c] = img_depth * ray_cross[r * 3 + c];
                }
            }
            Jpose_pix[9]  = z_ray_cross_pix[0];  Jpose_pix[10] = z_ray_cross_pix[1];  Jpose_pix[11] = z_ray_cross_pix[2];
            Jpose_pix[12] = z_ray_cross_pix[3];  Jpose_pix[13] = z_ray_cross_pix[4];  Jpose_pix[14] = z_ray_cross_pix[5];
            Jpose_pix[15] = z_ray_cross_pix[6];  Jpose_pix[16] = z_ray_cross_pix[7];  Jpose_pix[17] = z_ray_cross_pix[8];

            float temp_pix[18];
            for (int r = 0; r < 6; ++r) {
                for (int c = 0; c < 3; ++c) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; ++k) {
                        sum += Jpose_pix[r * 3 + k] * JtJ_cam_pix[k * 3 + c];
                    }
                    temp_pix[r * 3 + c] = sum;
                }
            }

            int idx = 0;
            for (int r = 0; r < 6; ++r) {
                for (int c = r; c < 6; ++c) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; ++k) {
                        sum += temp_pix[r * 3 + k] * Jpose_pix[c * 3 + k];
                    }
                    local_JtJ[idx] += sum;
                    idx++;
                }
            }

            for (int r = 0; r < 6; ++r) {
                float sum = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    sum += Jpose_pix[r * 3 + k] * Jtr_cam_pix[k];
                }
                local_Jtr[r] += sum;
            }

            float prod_alpha = 1.0f;
            float T = 1.0f;
            float3 acc_c = make_float3(0.0f, 0.0f, 0.0f);

            for (int base = 0; base < n_total; base += block_size) {
                int local_idx = base + tid;
                if (local_idx < n_total) {
                    uint32_t gid = indices[range.x + local_idx];
                    s_positions[tid] = positions_2d[gid];
                    s_invSigmas[tid] = inv_covariances_2d[gid];
                    s_colors[tid] = colors[gid];
                    s_alphas[tid] = alphas[gid];
                    s_depths[tid] = depths[gid];
                    s_pHats[tid] = p_hats[gid];
                }
                __syncthreads();

                int batch_count = min(block_size, n_total - base);
                for (int i = 0; i < batch_count && (i + base) < (int)n_contrib; ++i) {
                    float dx = s_positions[i].x - (float)x;
                    float dy = s_positions[i].y - (float)y;
                    float3 invSigma = s_invSigmas[i];
                    float v = invSigma.x * dx * dx + 2.0f * invSigma.y * dx * dy + invSigma.z * dy * dy;

                    float G = expf(-0.5f * v);
                    float alpha_i = fminf(0.99f, s_alphas[i] * G);
                    if (alpha_i < inv_255) {
                        continue;
                    }

                    float d = s_depths[i] + dx * s_pHats[i].x + dy * s_pHats[i].y;

                    float3 d_alpha = make_float3(s_colors[i].x * prod_alpha,
                                                 s_colors[i].y * prod_alpha,
                                                 s_colors[i].z * prod_alpha);

                    acc_c.x += alpha_i * d_alpha.x;
                    acc_c.y += alpha_i * d_alpha.y;
                    acc_c.z += alpha_i * d_alpha.z;

                    float inv_one_minus = 1.0f / (1.0f - alpha_i);
                    d_alpha.x -= (color.x - acc_c.x) * inv_one_minus;
                    d_alpha.y -= (color.y - acc_c.y) * inv_one_minus;
                    d_alpha.z -= (color.z - acc_c.z) * inv_one_minus;

                    float2 dl_mean2d = make_float2(invSigma.x * dx + invSigma.y * dy,
                                                   invSigma.y * dx + invSigma.z * dy);

                    float3 ray_g = make_float3((s_positions[i].x - intrinsics.c.x) / intrinsics.f.x,
                                               (s_positions[i].y - intrinsics.c.y) / intrinsics.f.y,
                                               1.0f);

                    float ray_cross_g[9];
                    ray_cross_g[0] = 0.0f;      ray_cross_g[1] = -ray_g.z;  ray_cross_g[2] = ray_g.y;
                    ray_cross_g[3] = ray_g.z;   ray_cross_g[4] = 0.0f;      ray_cross_g[5] = -ray_g.x;
                    ray_cross_g[6] = -ray_g.y;  ray_cross_g[7] = ray_g.x;   ray_cross_g[8] = 0.0f;

                    float inv_d = 1.0f / d;
                    float Jt[6];
                    Jt[0] = intrinsics.f.x * inv_d;
                    Jt[1] = 0.0f;
                    Jt[2] = 0.0f;
                    Jt[3] = intrinsics.f.y * inv_d;
                    Jt[4] = -intrinsics.f.x * ray_g.x * inv_d;
                    Jt[5] = -intrinsics.f.y * ray_g.y * inv_d;

                    float jt_dl[3];
                    jt_dl[0] = Jt[0] * dl_mean2d.x + Jt[1] * dl_mean2d.y;
                    jt_dl[1] = Jt[2] * dl_mean2d.x + Jt[3] * dl_mean2d.y;
                    jt_dl[2] = Jt[4] * dl_mean2d.x + Jt[5] * dl_mean2d.y;

                    float scale = alpha_i;
                    float Jt_cam[9];
                    Jt_cam[0] = jt_dl[0] * scale * d_alpha.x;
                    Jt_cam[1] = jt_dl[0] * scale * d_alpha.y;
                    Jt_cam[2] = jt_dl[0] * scale * d_alpha.z;
                    Jt_cam[3] = jt_dl[1] * scale * d_alpha.x;
                    Jt_cam[4] = jt_dl[1] * scale * d_alpha.y;
                    Jt_cam[5] = jt_dl[1] * scale * d_alpha.z;
                    Jt_cam[6] = jt_dl[2] * scale * d_alpha.x;
                    Jt_cam[7] = jt_dl[2] * scale * d_alpha.y;
                    Jt_cam[8] = jt_dl[2] * scale * d_alpha.z;

                    float JtJ_cam[9];
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            float sum = 0.0f;
                            for (int k = 0; k < 3; ++k) {
                                sum += Jt_cam[r * 3 + k] * Jt_cam[c * 3 + k];
                            }
                            JtJ_cam[r * 3 + c] = wc * sum;
                        }
                    }

                    float Jtr_cam[3];
                    Jtr_cam[0] = -wc * (Jt_cam[0] * color_error.x + Jt_cam[1] * color_error.y + Jt_cam[2] * color_error.z);
                    Jtr_cam[1] = -wc * (Jt_cam[3] * color_error.x + Jt_cam[4] * color_error.y + Jt_cam[5] * color_error.z);
                    Jtr_cam[2] = -wc * (Jt_cam[6] * color_error.x + Jt_cam[7] * color_error.y + Jt_cam[8] * color_error.z);

                    prod_alpha *= (1.0f - alpha_i);
                    if (T > 0.5f && prod_alpha <= 0.5f && img_depth > 0.5f) {
                        T = prod_alpha;
                        float ld = fabsf(depth_error);
                        float wd = (ld < depth_thresh) ? 1.0f : depth_thresh / ld;
                        wd /= img_depth;

                        JtJ_cam[0] += wd * ray_g.x * ray_g.x;
                        JtJ_cam[1] += wd * ray_g.x * ray_g.y;
                        JtJ_cam[2] += wd * ray_g.x * ray_g.z;
                        JtJ_cam[3] += wd * ray_g.y * ray_g.x;
                        JtJ_cam[4] += wd * ray_g.y * ray_g.y;
                        JtJ_cam[5] += wd * ray_g.y * ray_g.z;
                        JtJ_cam[6] += wd * ray_g.z * ray_g.x;
                        JtJ_cam[7] += wd * ray_g.z * ray_g.y;
                        JtJ_cam[8] += wd * ray_g.z * ray_g.z;

                        Jtr_cam[0] += wd * depth_error * ray_g.x;
                        Jtr_cam[1] += wd * depth_error * ray_g.y;
                        Jtr_cam[2] += wd * depth_error * ray_g.z;
                    }

                    float Jpose[18];
                    Jpose[0] = R[0][0];  Jpose[1] = R[0][1];  Jpose[2] = R[0][2];
                    Jpose[3] = R[1][0];  Jpose[4] = R[1][1];  Jpose[5] = R[1][2];
                    Jpose[6] = R[2][0];  Jpose[7] = R[2][1];  Jpose[8] = R[2][2];

                    float d_ray_cross[9];
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            d_ray_cross[r * 3 + c] = d * ray_cross_g[r * 3 + c];
                        }
                    }
                    Jpose[9]  = d_ray_cross[0];  Jpose[10] = d_ray_cross[1];  Jpose[11] = d_ray_cross[2];
                    Jpose[12] = d_ray_cross[3];  Jpose[13] = d_ray_cross[4];  Jpose[14] = d_ray_cross[5];
                    Jpose[15] = d_ray_cross[6];  Jpose[16] = d_ray_cross[7];  Jpose[17] = d_ray_cross[8];

                    float temp[18];
                    for (int r = 0; r < 6; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            float sum = 0.0f;
                            for (int k = 0; k < 3; ++k) {
                                sum += Jpose[r * 3 + k] * JtJ_cam[k * 3 + c];
                            }
                            temp[r * 3 + c] = sum;
                        }
                    }

                    int idx = 0;
                    for (int r = 0; r < 6; ++r) {
                        for (int c = r; c < 6; ++c) {
                            float sum = 0.0f;
                            for (int k = 0; k < 3; ++k) {
                                sum += temp[r * 3 + k] * Jpose[c * 3 + k];
                            }
                            local_JtJ[idx] += sum;
                            idx++;
                        }
                    }

                    for (int r = 0; r < 6; ++r) {
                        float sum = 0.0f;
                        for (int k = 0; k < 3; ++k) {
                            sum += Jpose[r * 3 + k] * Jtr_cam[k];
                        }
                        local_Jtr[r] += sum;
                    }

                    if (prod_alpha < 0.001f) {
                        break;
                    }
                }
                __syncthreads();
                if (prod_alpha < 0.001f) {
                    break;
                }
            }
        }
        
        // ========================================================================
        // FASE 6: Block Reduction (tree reduction)
        // ========================================================================
        
        // Shared memory para reduction
        __shared__ float shared_JtJ[TILE_SIZE * TILE_SIZE][21];
        __shared__ float shared_Jtr[TILE_SIZE * TILE_SIZE][6];
        
        // Copiar locales a shared
        for (int i = 0; i < 21; i++) shared_JtJ[tid][i] = local_JtJ[i];
        for (int i = 0; i < 6; i++) shared_Jtr[tid][i] = local_Jtr[i];
        __syncthreads();
        
        // Tree reduction
        for (int s = (TILE_SIZE * TILE_SIZE) / 2; s > 0; s >>= 1) {
            if (tid < s) {
                for (int i = 0; i < 21; i++) {
                    shared_JtJ[tid][i] += shared_JtJ[tid + s][i];
                }
                for (int i = 0; i < 6; i++) {
                    shared_Jtr[tid][i] += shared_Jtr[tid + s][i];
                }
            }
            __syncthreads();
        }
        
        // ========================================================================
        // FASE 7: Atomic Write a Memoria Global
        // ========================================================================
        
        if (tid == 0) {
            PoseOptimizationRgbdData &out = output_posedata[tile_idx];
            for (int i = 0; i < 21; i++) {
                atomicAdd(&out.JtJ[i], shared_JtJ[0][i]);
            }
            for (int i = 0; i < 6; i++) {
                atomicAdd(&out.Jtr[i], shared_Jtr[0][i]);
            }
        }
    }




    // ============================================================================
    // Funciones auxiliares
    // ============================================================================

    __device__ inline float evalGaussian2D(
        float dx, float dy,
        float inv_cov_xx, float inv_cov_yy, float inv_cov_xy)
    {
        // Calculamos el exponente de la Gaussiana 2D
        float exponente = -0.5f * (inv_cov_xx * dx * dx + inv_cov_yy * dy * dy + 2.0f * inv_cov_xy * dx * dy);
        // Devolvemos e^{exponente}
        return expf(exponente);
    }

    __device__ inline void invert2x2(
        float cov_xx, float cov_yy, float cov_xy,
        float &inv_cov_xx, float &inv_cov_yy, float &inv_cov_xy)
    {
        // Calculamos el determinante
        float det = cov_xx * cov_yy - cov_xy * cov_xy;
        if (det == 0.0f)
        {
            // Matriz singular, devolvemos identidad como fallback
            inv_cov_xx = 1.0f;
            inv_cov_yy = 1.0f;
            inv_cov_xy = 0.0f;
            return;
        }

        // Usamos la formula:
        // [ a  b ]^-1    =    1 / det * [ d  -b ]
        // [ c  d ]                      [ -c  a ]
        // nota: como cov es simetrica, b = c, por lo que la inversa es simetrica tambien

        float inv_det = 1.0f / det;

        // Invertimos la matriz 2x2
        inv_cov_xx =  cov_yy * inv_det;
        inv_cov_yy =  cov_xx * inv_det;
        inv_cov_xy = -cov_xy * inv_det;
    }

    // ============================================================================
    // Kernels de Covisibilidad
    // ============================================================================

    /**
     * @brief computeGaussiansVisibility_kernel
     * Implementación del kernel que marca gaussianas visibles en un frame
     * 
     * TODO: Completar la implementación siguiendo el algoritmo tile-based:
     * 1. Cada thread block procesa un tile (16x16 píxeles típicamente)
     * 2. Para cada pixel (x,y) del tile:
     *    - Iterar sobre gaussianas que intersectan el tile (almacenadas en ranges)
     *    - Evaluarlas como gaussianas 2D: alpha_i * exp(-0.5 * d^T * Σ^-1 * d)
     *    - Marcar como visible si contribuyen significativamente antes de saturación
     * 3. Guardar result: visibilities[gaussian_id] = 1 si es visible
     */
    __global__ void computeGaussiansVisibility_kernel(
        unsigned char *visibilities,
        const uint2 *ranges,
        const uint32_t *indices,
        const float3 *imgPositions,
        const float3 *imgInvSigmas,
        const float *alphas,
        uint2 numTiles,
        uint32_t width,
        uint32_t height)
    {
        // Calculamos coordenadas de pixel del hilo actual
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        bool inside = (x < (int)width && y < (int)height);

        // Obtenemos el rango de gaussianas para este tile
        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];
        int n = (int)(range.y - range.x);
        n = min(n, BLOCK_SIZE);

        // Shared memory para cargar gaussianas en batches
        __shared__ float3 s_positions[TILE_SIZE * TILE_SIZE];
        __shared__ float3 s_invSigmas[TILE_SIZE * TILE_SIZE];
        __shared__ float s_alphas[TILE_SIZE * TILE_SIZE];
        __shared__ uint32_t s_gids[TILE_SIZE * TILE_SIZE];

        int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
        int block_size = TILE_SIZE * TILE_SIZE;

        // Inicializamos transmitancia y flag de salida
        float T = 1.0f;
        bool done = !inside;

        // Procesamos gaussianas en batches para reducir lecturas globales
        for (int base = 0; base < n; base += block_size)
        {
            int local_idx = base + tid;
            if (local_idx < n)
            {
                uint32_t gid = indices[range.x + local_idx];
                s_gids[tid] = gid;
                s_positions[tid] = imgPositions[gid];
                s_invSigmas[tid] = imgInvSigmas[gid];
                s_alphas[tid] = alphas[gid];
            }
            __syncthreads();

            int batch_count = min(block_size, n - base);
            for (int i = 0; !done && i < batch_count; i++)
            {
                // Calculamos distancia de Mahalanobis para este pixel
                float dx = s_positions[i].x - (float)x;
                float dy = s_positions[i].y - (float)y;
                float v = s_invSigmas[i].x * dx * dx + 2.0f * s_invSigmas[i].y * dx * dy + s_invSigmas[i].z * dy * dy;

                // Calculamos alpha efectivo de la gaussiana
                float alpha_i = fminf(0.99f, s_alphas[i] * expf(-0.5f * v));

                // Filtramos contribuciones insignificantes
                if (alpha_i < (1.0f / 255.0f))
                {
                    continue;
                }

                // Marcamos visibilidad si contribuye
                visibilities[s_gids[i]] = 1;

                // Actualizamos transmitancia acumulada
                T *= (1.0f - alpha_i);

                // Cortamos si ya estamos suficientemente opacos
                if (T < 0.5f)
                {
                    done = true;
                }
            }

            __syncthreads();
        }
    }

    __global__ void computeGaussiansCovisibility_kernel(
        uint32_t *visibilityInter,
        uint32_t *visibilityUnion,
        unsigned char *visibilities1,
        unsigned char *visibilities2,
        uint32_t nbGaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= nbGaussians) return;

        unsigned char vis1 = visibilities1[idx];
        unsigned char vis2 = visibilities2[idx];

        if (vis1 | vis2) {
            atomicAdd(visibilityUnion, 1);
        }

        if (vis1 & vis2) {
            atomicAdd(visibilityInter, 1);
        }
    }

    // ============================================================================
    // Kernels de Gestión de Mapa: Prune, Outliers y Densificación
    // ============================================================================

    __global__ void pruneGaussians_kernel(
        uint32_t *nbRemoved,
        unsigned char *states,
        const float3 *scales,
        const float *alphas,
        float alphaThreshold,
        float scaleRatioThreshold,
        uint32_t nbGaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians) return;

        float alpha = alphas[idx];
        float3 scale = scales[idx];
        unsigned char state = 0;

        // criterio de opacidad (alfa < umbral)
        if (alpha < alphaThreshold) {
            state = 0xff;

        // criterio de escala (proporcion menor a escala, o escala muy pequeña)
        } else {
            float s_x = scale.x;
            float s_y = scale.y;
            float s_z = scale.z;

            float s_max = fmaxf(s_x, fmaxf(s_y, s_z));
            float s_min = fminf(s_x, fminf(s_y, s_z));
            float s_medio = s_x + s_y + s_z - s_max - s_min;

            if (s_medio / s_max < scaleRatioThreshold || s_max < 0.005f) {
                state = 0xff;
            }
        }

        if (state != 0) {
            atomicAdd(nbRemoved, 1);
        }

        states[idx] = state;

    }

    // Repasar
    __global__ void computeOutliers_kernel(
        float *outlierProb,
        float *totalAlpha,
        const uint2 *ranges,
        const uint32_t *indices,
        const float3 *imgPositions,
        const float3 *imgSigmas,
        const float3 *imgInvSigmas,
        const float2 *pHats,
        const float *depth,
        size_t depth_step,
        uint2 numTiles,
        uint32_t width,
        uint32_t height)
    {
        // Un thread por pixel
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        // Procesamos por batches
        int n = (int)(range.y - range.x);
        __shared__ float3 s_positions[TILE_SIZE * TILE_SIZE];
        __shared__ float3 s_sigmas[TILE_SIZE * TILE_SIZE];
        __shared__ float3 s_invSigmas[TILE_SIZE * TILE_SIZE];
        __shared__ float2 s_pHats[TILE_SIZE * TILE_SIZE];
        __shared__ uint32_t s_gids[TILE_SIZE * TILE_SIZE];
        __shared__ float s_alphas[TILE_SIZE * TILE_SIZE];

        int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
        int block_size = TILE_SIZE * TILE_SIZE;
        float px = (float)x;
        float py = (float)y;

        // Leer profundidad observada
        const char *depth_row = reinterpret_cast<const char *>(depth) + y * depth_step;
        float depth_obs = *(reinterpret_cast<const float *>(depth_row) + x);

        // Procesamos gaussianas en batches para reducir lecturas globales
        for (int base = 0; base < n; base += block_size)
        {
            int local_idx = base + tid;
            if (local_idx < n)
            {
                uint32_t gid = indices[range.x + local_idx];
                s_gids[tid] = gid;
                s_positions[tid] = imgPositions[gid];
                s_sigmas[tid] = imgSigmas[gid];
                s_invSigmas[tid] = imgInvSigmas[gid];
                s_pHats[tid] = pHats[gid];
                // Nota: necesitaremos alphas, se asume que se pasan o se recuperan
            }
            __syncthreads();

            int batch_count = min(block_size, n - base);
            for (int i = 0; i < batch_count; i++)
            {
                float3 pos = s_positions[i];
                float2 pHat = s_pHats[i];
                uint32_t gid = s_gids[i];

                // Calcular desplazamiento en imagen
                float dx = px - pos.x;
                float dy = py - pos.y;

                // Calcular profundidad renderizada: d = z + pHat·(x-u, y-v)
                float depth_rendered = pos.z + pHat.x * dx + pHat.y * dy;

                // Evaluar gaussiana 2D para obtener alpha_i
                float3 invSigma = s_invSigmas[i];
                float gauss_val = evalGaussian2D(dx, dy, invSigma.x, invSigma.z, invSigma.y);
                // Nota: alpha_i tendría que multiplicarse por alphas[gid], pero no lo tenemos en shared
                // Por ahora usamos gauss_val directamente
                float alpha_i = gauss_val;

                if (alpha_i < (1.0f / 255.0f) || depth_rendered <= 0.0f)
                {
                    continue;
                }

                // Acumular en totalAlpha
                atomicAdd(&totalAlpha[gid], alpha_i);

                // Detectar outliers: si hay mismatch significativo entre observado y renderizado
                if (depth_obs > 0.1f && depth_rendered < 0.8f * depth_obs)
                {
                    // Hay oclusión/mismatch: profundidad renderizada es mucho menor que observada
                    atomicAdd(&outlierProb[gid], alpha_i);
                }
            }

            __syncthreads();
        }
    }

    __global__ void removeOutliers_kernel(
        uint32_t *nbRemoved,
        unsigned char *states,
        const float *outlierProb,
        const float *totalAlpha,
        float threshold,
        uint32_t nbGaussians)
    {
        // Un thread por gaussiana
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians) return;

        unsigned char state = 0;

        if (totalAlpha[idx] > 1.0f && outlierProb[idx] / totalAlpha[idx] > threshold)
        {
            atomicAdd(nbRemoved, 1);
            state = 0xff;
        }

        states[idx] = state;

    }

    __global__ void computeDensityMask_kernel(
        float *maskData,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        const float *__restrict__ depth,
        size_t depth_step,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        size_t mask_step)
    {
        // Un thread por pixel
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        bool inside = (x < (int)width && y < (int)height);

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];
        int n = (int)(range.y - range.x);

        const char *depth_row = reinterpret_cast<const char *>(depth) + y * depth_step;
        float depth_obs = *(reinterpret_cast<const float *>(depth_row) + x);

        float prod_alpha = 1.0f;
        float depth_rendered = 0.0f;
        float T = 1.0f;
        // Si no hay profundidad valida, no hacemos el recorrido completo
        bool done = (depth_obs < 0.5f) || !inside;
        

        // Shared memory para cargar gaussianas en batches
        __shared__ float3 s_positions[TILE_SIZE * TILE_SIZE];
        __shared__ float3 s_invSigmas[TILE_SIZE * TILE_SIZE];
        __shared__ float2 s_pHats[TILE_SIZE * TILE_SIZE];
        __shared__ float3 s_colors[TILE_SIZE * TILE_SIZE];
        __shared__ float s_alphas[TILE_SIZE * TILE_SIZE];

        // Cargamos por batches
        for (int base = 0; base < n; base += TILE_SIZE * TILE_SIZE)
        {
            int local_idx = base + threadIdx.y * TILE_SIZE + threadIdx.x;
            if (local_idx < n)
            {
                uint32_t gid = indices[range.x + local_idx];
                s_positions[threadIdx.y * TILE_SIZE + threadIdx.x] = imgPositions[gid];
                s_invSigmas[threadIdx.y * TILE_SIZE + threadIdx.x] = imgInvSigmas[gid];
                s_pHats[threadIdx.y * TILE_SIZE + threadIdx.x] = pHats[gid];
                s_colors[threadIdx.y * TILE_SIZE + threadIdx.x] = colors[gid];
                s_alphas[threadIdx.y * TILE_SIZE + threadIdx.x] = alphas[gid];
            }
            __syncthreads();

            int batch_count = min(TILE_SIZE * TILE_SIZE, n - base);
            for (int i = 0; !done && i < batch_count; i++)
            {
                float dx = s_positions[i].x - (float)x;
                float dy = s_positions[i].y - (float)y;
                float3 invSigma = s_invSigmas[i];
                float v = invSigma.x * dx * dx + 2.0f * invSigma.y * dx * dy + invSigma.z * dy * dy;
                float alpha_i = fminf(0.99f, s_alphas[i] * expf(-0.5f * v));

                if (alpha_i < (1.0f / 255.0f))
                {
                    continue;
                }

                // Acumulamos transmitancia a lo largo de las gaussianas visibles
                prod_alpha *= (1.0f - alpha_i);

                if (T > 0.5f && prod_alpha < 0.5f)
                {
                    // Guardamos la profundidad mediana al cruzar 0.5
                    T = prod_alpha;
                    depth_rendered = s_positions[i].z + s_pHats[i].x * dx + s_pHats[i].y * dy;
                }

                if (prod_alpha < 0.0001f)
                {
                    done = true;
                    break;
                }
            }

            __syncthreads();
        }

        // Decidimos el valor final del mask segun profundidad observada y transmitancia
        float val = 0.0f;
        if (depth_obs > 0.5f)
        {
            if (prod_alpha > 0.5f)
            {
                val = prod_alpha;
            }
            else
            {
                // Si la profundidad renderizada queda por delante con gran error, marcamos densificacion
                float depth_error = depth_rendered - depth_obs;
                if (depth_obs < depth_rendered && depth_error > 0.2f * depth_obs)
                {
                    val = 1.0f;
                }
            }
        }

        // Escribimos el mask respetando el stride en bytes
        size_t mask_idx = y * (mask_step / sizeof(float)) + x;
        maskData[mask_idx] = val;
    }

    __global__ void densifyGaussians_kernel(
        float3 *positions,
        float3 *scales,
        float4 *orientations,
        float3 *colors,
        float *alphas,
        uint32_t *instanceCounter,
        const uchar3 *rgb,
        size_t rgb_step,
        const float *depth,
        size_t depth_step,
        const float4 *normals,
        size_t normals_step,
        const float *mask,
        size_t mask_step,
        CameraPose cameraPose,
        IntrinsicParameters intrinsics,
        uint32_t sample_dx,
        uint32_t sample_dy,
        uint32_t width,
        uint32_t height,
        uint32_t maxGaussians)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (sample_dx > 1)
        {
            // Trabajamos por celdas de muestreo
            int u_min = x * (int)sample_dx;
            int v_min = y * (int)sample_dy;

            if (u_min >= (int)width || v_min >= (int)height)
            {
                return;
            }

            // Acumulamos pixeles marcados por el mask
            float3 img_pos = make_float3(0.0f, 0.0f, 0.0f);
            float3 rgb_acc = make_float3(0.0f, 0.0f, 0.0f);
            int n = 0;

            for (int u = u_min; u < u_min + (int)sample_dx && u < (int)width; u++)
            {
                for (int v = v_min; v < v_min + (int)sample_dy && v < (int)height; v++)
                {
                    const char *mask_row = reinterpret_cast<const char *>(mask) + v * mask_step;
                    float mask_val = *(reinterpret_cast<const float *>(mask_row) + u);
                    if (mask_val > 0.0f)
                    {
                        // Leemos depth y color
                        const char *depth_row = reinterpret_cast<const char *>(depth) + v * depth_step;
                        float d = *(reinterpret_cast<const float *>(depth_row) + u);
                        if (!(d > 0.0f))
                        {
                            continue;
                        }

                        const unsigned char *rgb_row = reinterpret_cast<const unsigned char *>(rgb) + v * rgb_step;
                        const uchar3 *rgb_row3 = reinterpret_cast<const uchar3 *>(rgb_row);
                        uchar3 bgr = rgb_row3[u];

                        img_pos.x += (float)u;
                        img_pos.y += (float)v;
                        img_pos.z += d;
                        rgb_acc.x += bgr.z / 255.0f;
                        rgb_acc.y += bgr.y / 255.0f;
                        rgb_acc.z += bgr.x / 255.0f;
                        n++;
                    }
                }
            }

            if (n < ((sample_dx * sample_dy) >> 1))
            {
                return;
            }

            // Reservamos un nuevo indice de gaussiana
            uint32_t idx = atomicAdd(instanceCounter, 1u);
            if (idx >= maxGaussians)
            {
                return;
            }

            // Promediamos posicion e intensidad
            float inv_n = 1.0f / (float)n;
            img_pos.x *= inv_n;
            img_pos.y *= inv_n;
            img_pos.z *= inv_n;
            rgb_acc.x *= inv_n;
            rgb_acc.y *= inv_n;
            rgb_acc.z *= inv_n;

            // Reproyectamos a camara y luego a mundo
            float3 pos_cam = make_float3(
                img_pos.z * (img_pos.x - intrinsics.c.x) / intrinsics.f.x,
                img_pos.z * (img_pos.y - intrinsics.c.y) / intrinsics.f.y,
                img_pos.z);

            float3 pos_world = cameraPose.position + rotateByQuaternion(cameraPose.orientation, pos_cam);
            positions[idx] = pos_world;

            // Escala inicial segun tamano de celda
            float scale_x = img_pos.z * (float)sample_dx / intrinsics.f.x;
            float scale_y = img_pos.z * (float)sample_dy / intrinsics.f.y;
            float s = 0.5f * (scale_x + scale_y);
            scales[idx] = make_float3(s, s, 0.1f * s);

            // Orientacion desde normal (si existe)
            float3 normal = make_float3(0.0f, 0.0f, 1.0f);
            if (normals && normals_step > 0)
            {
                int nu = min((int)width - 1, max(0, (int)(img_pos.x + 0.5f)));
                int nv = min((int)height - 1, max(0, (int)(img_pos.y + 0.5f)));
                const char *normal_row = reinterpret_cast<const char *>(normals) + nv * normals_step;
                const float4 *normal_row4 = reinterpret_cast<const float4 *>(normal_row);
                float4 n4 = normal_row4[nu];
                normal = make_float3(n4.x, n4.y, n4.z);
            }
            if (normal.z < 0.0f)
            {
                normal.x = -normal.x;
                normal.y = -normal.y;
                normal.z = -normal.z;
            }

            float4 q = quatFromTwoVectors(make_float3(0.0f, 0.0f, 1.0f), normal);
            float4 q_world = quatMultiply(cameraPose.orientation, q);
            orientations[idx] = q_world;

            // Guardamos color y opacidad
            colors[idx] = rgb_acc;
            alphas[idx] = 1.0f;
        }
        else
        {
            // Modo pixel a pixel
            if (x >= (int)width || y >= (int)height)
            {
                return;
            }

            const char *mask_row = reinterpret_cast<const char *>(mask) + y * mask_step;
            float mask_val = *(reinterpret_cast<const float *>(mask_row) + x);
            if (mask_val <= 0.0f)
            {
                return;
            }

            // Leemos depth y color
            const char *depth_row = reinterpret_cast<const char *>(depth) + y * depth_step;
            float d = *(reinterpret_cast<const float *>(depth_row) + x);
            if (!(d > 0.0f))
            {
                return;
            }

            const unsigned char *rgb_row = reinterpret_cast<const unsigned char *>(rgb) + y * rgb_step;
            const uchar3 *rgb_row3 = reinterpret_cast<const uchar3 *>(rgb_row);
            uchar3 bgr = rgb_row3[x];
            float3 rgb_val = make_float3(bgr.z / 255.0f, bgr.y / 255.0f, bgr.x / 255.0f);

            // Reservamos un nuevo indice de gaussiana
            uint32_t idx = atomicAdd(instanceCounter, 1u);
            if (idx >= maxGaussians)
            {
                return;
            }

            // Reproyectamos a camara y luego a mundo
            float3 pos_cam = make_float3(
                d * ((float)x - intrinsics.c.x) / intrinsics.f.x,
                d * ((float)y - intrinsics.c.y) / intrinsics.f.y,
                d);

            float3 pos_world = cameraPose.position + rotateByQuaternion(cameraPose.orientation, pos_cam);
            positions[idx] = pos_world;

            // Escala inicial para un solo pixel
            float scale_x = 0.8f * d * (float)sample_dx / intrinsics.f.x;
            float scale_y = 0.8f * d * (float)sample_dy / intrinsics.f.y;
            float s = 0.5f * (scale_x + scale_y);
            scales[idx] = make_float3(s, s, 0.1f * s);

            // Orientacion desde normal (si existe)
            float3 normal = make_float3(0.0f, 0.0f, 1.0f);
            if (normals && normals_step > 0)
            {
                const char *normal_row = reinterpret_cast<const char *>(normals) + y * normals_step;
                const float4 *normal_row4 = reinterpret_cast<const float4 *>(normal_row);
                float4 n4 = normal_row4[x];
                normal = make_float3(n4.x, n4.y, n4.z);
            }
            if (normal.z < 0.0f)
            {
                normal.x = -normal.x;
                normal.y = -normal.y;
                normal.z = -normal.z;
            }

            float4 q = quatFromTwoVectors(make_float3(0.0f, 0.0f, 1.0f), normal);
            float4 q_world = quatMultiply(cameraPose.orientation, q);
            orientations[idx] = q_world;

            // Guardamos color y opacidad
            colors[idx] = rgb_val;
            alphas[idx] = 1.0f;
        }
    }

    // ============================================================================
    // Optimizacion de keyframes
    // ============================================================================

    __global__ void perTileBucketCount(
        uint32_t* __restrict__ bucketCount,
        const uint2* __restrict__ tileRanges,
        int numTiles)
    {
        // Un thread por tile
        int tileId = blockIdx.x * blockDim.x + threadIdx.x;
        if (tileId >= numTiles) return;

        // Contamos las gaussianas en el tile
        uint2 range = tileRanges[tileId];
        int amount = range.y - range.x;

        // Agrupamos en buckets de tamaño 32
        constexpr uint32_t BUCKET_SIZE = 32;
        bucketCount[tileId] = (amount + BUCKET_SIZE - 1) / BUCKET_SIZE;
    } 

    __global__ void optimizeGaussiansForwardPass(
        const uint2 *ranges,
        const uint32_t *indices,
        const float2 *positions_2d,
        const float3 *inv_covariances_2d,
        const float2 *p_hats,
        const float *depths,
        const float3 *colors,
        const float *alphas,
        //const uint32_t *per_tile_buckets, Uso la version con prefix sum
        const uint32_t *bucketOffsets, // Version de prefix sum de per_tile_buckets
        uint32_t *bucket_to_tile,
        float *sampled_T,
        float3 *sampled_ar,
        float *final_T,
        uint32_t *n_contrib,
        uint32_t *max_contrib,
        float3 *output_color,
        float *output_depth,
        float3 *color_error,
        float *depth_error,
        const float3 *observed_rgb,
        const float *observed_depth,
        float3 bg_color,
        uint2 num_tiles,
        int width,
        int height)
    {
        // Un thread por pixel
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int tId = threadIdx.x + threadIdx.y * blockDim.x;

        bool inside = (x < (int)width && y < (int)height);
        bool done = !inside;

        // Obtenemos el tile y rango de gaussianas para este pixel
        int tileId = blockIdx.y * num_tiles.x + blockIdx.x;
        uint2 range = ranges[tileId];
        int n = (int)(range.y - range.x);

        // Completamos la data de bucketToTile
        uint32_t bucket_base = (tileId == 0) ? 0 : bucketOffsets[tileId - 1];

        int bucketCount = (n + 31) / 32;
        for (int i = 0; i < (bucketCount + TILE_SIZE - 1) / TILE_SIZE; ++i)
        {
            int bIdx = i * TILE_SIZE + tId;
            
            if (bIdx < bucketCount) bucketToTile[bucket_base + bIdx] = tileId;
        }

        // Inicializamos params
        float T = 1.0f;
        float3 color = make_float3(0.0f, 0.0f, 0.0f);
        float depth = 0.0f;
        uint32_t contributor = 0;
        uint32_t last_contributor = 0;

        // Procesamos por batches
        __shared__ float2 s_positions_2d[TILE_SIZE * TILE_SIZE];
        __shared__ float3 s_inv_covariances_2d[TILE_SIZE * TILE_SIZE];
        __shared__ float2 s_p_hats[TILE_SIZE * TILE_SIZE];
        __shared__ float3 s_colors[TILE_SIZE * TILE_SIZE];
        __shared__ float3 s_depths[TILE_SIZE * TILE_SIZE];
        __shared__ float s_alphas[TILE_SIZE * TILE_SIZE];
        __shared__ uint32_t s_gids[TILE_SIZE * TILE_SIZE];

        int n_batches = (n + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE);
        for (int batch = 0; batch < n_batches && !done; batch++)
        {
            int local_idx = batch * TILE_SIZE * TILE_SIZE + threadIdx.y * TILE_SIZE + threadIdx.x;
            if (local_idx < n)
            {
                uint32_t gid = indices[range.x + local_idx];
                s_gids[threadIdx.y * TILE_SIZE + threadIdx.x] = gid;
                s_positions_2d[threadIdx.y * TILE_SIZE + threadIdx.x] = positions_2d[gid];
                s_inv_covariances_2d[threadIdx.y * TILE_SIZE + threadIdx.x] = inv_covariances_2d[gid];
                s_p_hats[threadIdx.y * TILE_SIZE + threadIdx.x] = p_hats[gid];
                s_colors[threadIdx.y * TILE_SIZE + threadIdx.x] = colors[gid];
                s_depths[threadIdx.y * TILE_SIZE + threadIdx.x] = depths[gid];
                s_alphas[threadIdx.y * TILE_SIZE + threadIdx.x] = alphas[gid];
            }
            __syncthreads();

            int batch_count = min(TILE_SIZE * TILE_SIZE, n - batch * TILE_SIZE * TILE_SIZE);
            for (int i = 0; i < batch_count && !done; i++)
            {
                int global_idx = batch * (TILE_SIZE * TILE_SIZE) + i;
                if (global_idx % 32 == 0)
                {
                    uint32_t bucket_idx = global_idx / 32;
                    int sampleIdx = (bucket_base + bucket_idx) * BLOCK_SIZE + tId;
                    sampled_T[sampleIdx] = T;
                    sampled_ar[sampleIdx] = color;
                }

                contributor++;

                float2 pos = s_positions_2d[i];
                float3 inv_cov = s_inv_covariances_2d[i];
                float2 p_hat = s_p_hats[i];
                float3 col = s_colors[i];
                float z_c = s_depths[i];
                float alpha = s_alphas[i];

                float T_pre = T;
                // Calculamos contribución de esta gaussiana al pixel (x,y)
                // usando la formula: alpha_i * exp(-0.5 * d^T * Σ^-1 * d), con d = (x-u, y-v)
                float dx = (float)x - pos.x;
                float dy = (float)y - pos.y;

                float v = inv_cov.x * dx * dx + 2.0f * inv_cov.y * dx * dy + inv_cov.z * dy * dy;
                float gauss_val = expf(-0.5f * v);
                float contrib_alpha = fminf(0.99f, alpha * gauss_val);

                if (contrib_alpha < (1.0f / 255.0f) || v <= 0.0f)
                {
                    continue;
                }

                color += T * contrib_alpha * col;
                float test_T = T * (1.0f - contrib_alpha);

                if (T_pre > 0.5f && test_T <= 0.5f)
                {
                    depth = z_c + p_hat.x * dx + p_hat.y * dy;
                }

                T = test_T;
                last_contributor = contributor;

                if (test_T < 0.0001f)
                {
                    done = true;
                    continue;
                }
            }
            __syncthreads();
        }

        if (inside)
        {
            int pix_id = y * width + x;
            final_T[pix_id] = T;
            n_contrib[pix_id] = last_contributor;

            color += T * bg_color;
            output_color[pix_id] = color;
            output_depth[pix_id] = depth;

            float3 obs_color = observed_rgb[pix_id];
            color_error[pix_id] = make_float3(color.x - obs_color.x,
                                              color.y - obs_color.y,
                                              color.z - obs_color.z);

            float img_depth = observed_depth[pix_id];
            depth_error[pix_id] = img_depth;

            // Se actualiza con reduccion por bloque mas abajo.
        }

        typedef cub::BlockReduce<uint32_t, TILE_SIZE, cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                 TILE_SIZE>
            BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        last_contributor = BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
        if (tId == 0)
        {
            max_contrib[tileId] = last_contributor;
        }
    }

    __global__ void optimizeGaussiansPerGaussianPass(
        const uint2 *ranges,
        const uint32_t *indices,
        const float2 *positions_2d,
        const float3 *inv_covariances_2d,
        const float2 *p_hats,
        const float *depths,
        const float3 *colors,
        const float *alphas,
        //const uint32_t *per_tile_buckets,
        const uint32_t *bucketOffsets, // Version prefix sum de per_tile_buckets
        const uint32_t *bucket_to_tile,
        const float *sampled_T,
        const float3 *sampled_ar,
        const uint32_t *n_contrib,
        const uint32_t *max_contrib,
        const float3 *output_color,
        const float *output_depth,
        const float3 *color_error,
        const float *depth_error,
        DeltaGaussian2D *delta_gaussians,
        float w_depth,
        float w_dist,
        uint2 num_tiles,
        int width,
        int height,
        int num_buckets)
    {
        // Mapeamos hilos a warps para procesar buckets
        uint32_t warp_id = threadIdx.x >> 5;
        uint32_t lane_id = threadIdx.x & 31;
        uint32_t warps_per_block = blockDim.x >> 5;

        // Cada warp procesa un bucket global
        uint32_t global_bucket_id =
            blockIdx.x * warps_per_block + warp_id;

        if (global_bucket_id >= num_buckets)
            return;

        // Resolvemos el tile dueño del bucket
        uint32_t tileId = bucket_to_tile[global_bucket_id];
        uint2 range = ranges[tileId];

        // bucket base (prefix sum)
        uint32_t bucket_base =
            (tileId == 0) ? 0 : bucketOffsets[tileId - 1];

        uint32_t bucket_idx = global_bucket_id - bucket_base;

        // si el bucket cae despues de max_contrib no aporta
        if (bucket_idx * 32 >= max_contrib[tileId])
            return;

        // cada lane apunta a una gaussiana dentro del tile
        uint32_t splat_idx_in_tile = bucket_idx * 32 + lane_id;
        bool valid = splat_idx_in_tile < (range.y - range.x);

        uint32_t gId;
        if (valid)
        {
            gId = indices[range.x + splat_idx_in_tile];
        }

        uint32_t num_splats_in_tile = min((uint32_t)(range.y - range.x), (uint32_t)BLOCK_SIZE);
        bool valid_splat = valid && (splat_idx_in_tile < num_splats_in_tile);

        float2 g_pos2 = make_float2(0.0f, 0.0f);
        float3 g_pos = make_float3(0.0f, 0.0f, 0.0f);
        float3 g_inv_sigma = make_float3(0.0f, 0.0f, 0.0f);
        float3 g_color = make_float3(0.0f, 0.0f, 0.0f);
        float g_alpha = 0.0f;
        float2 g_p_hat = make_float2(0.0f, 0.0f);

        // Cargamos parametros de la gaussiana
        if (valid_splat)
        {
            g_pos2 = positions_2d[gId];
            g_pos = make_float3(g_pos2.x, g_pos2.y, depths[gId]);
            g_inv_sigma = inv_covariances_2d[gId];
            g_color = colors[gId];
            g_alpha = alphas[gId];
            g_p_hat = p_hats[gId];
        }

        // Acumulador local de gradientes para esta gaussiana 2d
        DeltaGaussian2D delta;
        delta.meanImg = make_float2(0.0f, 0.0f);
        delta.invSigmaImg = make_float3(0.0f, 0.0f, 0.0f);
        delta.color = make_float3(0.0f, 0.0f, 0.0f);
        delta.depth = 0.0f;
        delta.alpha = 0.0f;
        delta.pHat = make_float2(0.0f, 0.0f);
        delta.n = 0;

        // Esquina superior izquierda del tile en pixeles
        uint2 tile = make_uint2(tileId % num_tiles.x, tileId / num_tiles.x);
        uint2 pix_min = make_uint2(tile.x * TILE_SIZE, tile.y * TILE_SIZE);

        float T = 0.0f;
        float last_contributor = 0.0f;
        float3 acc_c = make_float3(0.0f, 0.0f, 0.0f);
        float3 col_err = make_float3(0.0f, 0.0f, 0.0f);
        float3 color = make_float3(0.0f, 0.0f, 0.0f);
        float img_depth = 0.0f;
        float depth = 0.0f;

        const unsigned mask = 0xffffffffu;

        // Recorremos el bloque expandido para shuffles por warp
        for (int i = 0; i < BLOCK_SIZE + 31; ++i)
        {
            T = __shfl_up_sync(mask, T, 1);
            last_contributor = __shfl_up_sync(mask, last_contributor, 1);
            acc_c.x = __shfl_up_sync(mask, acc_c.x, 1);
            acc_c.y = __shfl_up_sync(mask, acc_c.y, 1);
            acc_c.z = __shfl_up_sync(mask, acc_c.z, 1);
            color.x = __shfl_up_sync(mask, color.x, 1);
            color.y = __shfl_up_sync(mask, color.y, 1);
            color.z = __shfl_up_sync(mask, color.z, 1);
            col_err.x = __shfl_up_sync(mask, col_err.x, 1);
            col_err.y = __shfl_up_sync(mask, col_err.y, 1);
            col_err.z = __shfl_up_sync(mask, col_err.z, 1);
            img_depth = __shfl_up_sync(mask, img_depth, 1);
            depth = __shfl_up_sync(mask, depth, 1);

            int idx = i - (int)lane_id;
            int pix_x = (int)pix_min.x + idx % TILE_SIZE;
            int pix_y = (int)pix_min.y + idx / TILE_SIZE;
            uint32_t pix_id = (uint32_t)(width * pix_y + pix_x);
            bool valid_pixel = (idx >= 0) && (idx < BLOCK_SIZE) && (pix_x < width) && (pix_y < height) &&
                               (pix_x >= 0) && (pix_y >= 0);

            // Lane 0 carga el estado muestreado del bucket
            if (valid_splat && valid_pixel && lane_id == 0 && idx < BLOCK_SIZE)
            {
                T = sampled_T[global_bucket_id * BLOCK_SIZE + idx];
                acc_c = sampled_ar[global_bucket_id * BLOCK_SIZE + idx];
                color = output_color[pix_id];
                depth = output_depth[pix_id];
                last_contributor = (float)n_contrib[pix_id];
                col_err = color_error[pix_id];
                img_depth = depth_error[pix_id];
            }

            // Computamos gradientes para este pixel
            if (valid_splat && valid_pixel && 0 <= idx && idx < BLOCK_SIZE)
            {
                if (splat_idx_in_tile >= (uint32_t)last_contributor)
                    continue;

                // Distancia en imagen para evaluar la gaussiana
                float dx = g_pos.x - (float)pix_x;
                float dy = g_pos.y - (float)pix_y;
                float v = g_inv_sigma.x * dx * dx + 2.0f * g_inv_sigma.y * dx * dy + g_inv_sigma.z * dy * dy;
                float G = expf(-0.5f * v);
                float alpha = fminf(0.99f, g_alpha * G);
                if (alpha < (1.0f / 255.0f))
                    continue;

                // dC/dalpha = T * color - (C - C_{<=n}) / (1 - alpha)
                float3 d_alpha = make_float3(g_color.x * T, g_color.y * T, g_color.z * T);
                acc_c.x += d_alpha.x * alpha;
                acc_c.y += d_alpha.y * alpha;
                acc_c.z += d_alpha.z * alpha;
                float inv_one_minus = 1.0f / (1.0f - alpha);
                d_alpha.x -= (color.x - acc_c.x) * inv_one_minus;
                d_alpha.y -= (color.y - acc_c.y) * inv_one_minus;
                d_alpha.z -= (color.z - acc_c.z) * inv_one_minus;

                // dL/dalpha = dL/dC * dC/dalpha, multiplicamos por dL/dC = -color_error
                d_alpha.x *= -col_err.x;
                d_alpha.y *= -col_err.y;
                d_alpha.z *= -col_err.z;

                // dL/dalpha = dL/dC * dC/dalpha
                float dl_alpha = d_alpha.x + d_alpha.y + d_alpha.z;
                float a_G_dl = alpha * dl_alpha;

                delta.n++;

                // dL/dc = -color_error * alpha * T
                delta.color.x -= alpha * T * col_err.x;
                delta.color.y -= alpha * T * col_err.y;
                delta.color.z -= alpha * T * col_err.z;
                delta.alpha -= a_G_dl;

                // dL/dmu_{2D} = dL/dalpha * alpha * (Sigma^{-1} * (x - mu_{2D})) 
                delta.meanImg.x -= a_G_dl * (g_inv_sigma.x * dx + g_inv_sigma.y * dy);
                delta.meanImg.y -= a_G_dl * (g_inv_sigma.y * dx + g_inv_sigma.z * dy);

                // dL/dInvSigma_{2D} = -0.5 * dL/dalpha * alpha * delta * delta^T
                delta.invSigmaImg.x -= 0.5f * a_G_dl * dx * dx;
                delta.invSigmaImg.y -= 0.5f * a_G_dl * dx * dy;
                delta.invSigmaImg.z -= 0.5f * a_G_dl * dy * dy;

                // Actualizamos transmitancia para el siguiente splat
                float test_T = T * (1.0f - alpha);
                if (T > 0.5f && test_T <= 0.5f)
                {
                    float depth_err = (img_depth > 0.1f) ? (depth - img_depth) : 0.0f;
                    delta.depth -= w_depth * depth_err;
                    delta.meanImg.x -= w_depth * depth_err * g_p_hat.x;
                    delta.meanImg.y -= w_depth * depth_err * g_p_hat.y;
                    delta.pHat.x -= w_depth * depth_err * dx;
                    delta.pHat.y -= w_depth * depth_err * dy;
                }

                float di = g_pos.z + dx * g_p_hat.x + dy * g_p_hat.y;
                float dd = di - depth;
                float dist_coeff = w_dist * alpha * T * dd;
                delta.depth -= dist_coeff;
                delta.meanImg.x -= dist_coeff * g_p_hat.x;
                delta.meanImg.y -= dist_coeff * g_p_hat.y;
                delta.pHat.x -= dist_coeff * dx;
                delta.pHat.y -= dist_coeff * dy;

                T = test_T;
            }
        }

        if (valid_splat && delta.n > 0)
        {
            atomicAdd(&delta_gaussians[gId].depth, delta.depth);
            atomicAdd(&delta_gaussians[gId].pHat.x, delta.pHat.x);
            atomicAdd(&delta_gaussians[gId].pHat.y, delta.pHat.y);
            atomicAdd(&delta_gaussians[gId].meanImg.x, delta.meanImg.x);
            atomicAdd(&delta_gaussians[gId].meanImg.y, delta.meanImg.y);
            atomicAdd(&delta_gaussians[gId].invSigmaImg.x, delta.invSigmaImg.x);
            atomicAdd(&delta_gaussians[gId].invSigmaImg.y, delta.invSigmaImg.y);
            atomicAdd(&delta_gaussians[gId].invSigmaImg.z, delta.invSigmaImg.z);
            atomicAdd(&delta_gaussians[gId].color.x, delta.color.x);
            atomicAdd(&delta_gaussians[gId].color.y, delta.color.y);
            atomicAdd(&delta_gaussians[gId].color.z, delta.color.z);
            atomicAdd(&delta_gaussians[gId].alpha, delta.alpha);
            atomicAdd(&delta_gaussians[gId].n, delta.n);
        }
    }

    __global__ void computeDeltaGaussians3D_kernel(
        DeltaGaussian3D *delta_gaussians_3d,
        const float3 *positions,
        const float3 *scales,
        const float4 *orientations,
        const float3 *colors,
        const float *alphas,
        const DeltaGaussian2D *delta_gaussians_2d, // Grad de color, depth, media2d, cov2d, pHat, depth
        CameraPose camera_pose,
        IntrinsicParameters intrinsics,
        float lambda_iso,
        int n_gaussians)
    {
        // Un thread por gaussiana
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_gaussians) {
            return;
        }

        // Preparamos los deltas
        DeltaGaussian2D delta_2d = delta_gaussians_2d[idx];
        DeltaGaussian3D delta_3d;
        
        if (delta_2d.n == 0)
        {
            delta_gaussians_3d[idx].n = 0; // Caso borde para gaussiana sin contribucion
            return;
        }

        delta_3d.n = delta_2d.n;

        // Regularizamos el learning rate para que las gaussianas con mayor contribucion
        // se vean menos afectadas
        const float eta_n = 1.0f / (5.0f + float(delta_2d.n));



        // Lista de gradientes a calcular:
        // Color: dL/dc_n = alpha_n * T_n
        // Opacidad: dL/dalpha_n = c_n * T_n - S_n
        // Media: dL/du_n
        // d

        // Color: dL/dc_n = eta_n * c_n
        // Opacidad: dL/dalpha_n = eta_n * alpha_n
        // Covarianza: dL/dSigma_n = - 0.5 * Sigma'^{-1} * delta_n * delta_n^T * Sigma'^{-1} (con delta_n = (x-u, y-v))
        // Media: dL/dmu = antiproyectar

        // Pose: dL/dt = dL/dmu * dmu/dt + dL/dSigma * dSigma/dt
        // Rotacion: dL/dR = dL/dM * S^T
        // escala: dL/dS = R^T dL/dM
        
        // Gradiente isotropico: dL/dSigma_iso = lambda_iso * (sigma_x - sigma_y)

        // ============================================================================
        // Gradiente de color y opacidad:
        // ============================================================================
        delta_3d.color = delta_2d.color * eta_n;
        delta_3d.alpha = delta_2d.alpha * eta_n;

        // ============================================================================
        // Gradientes de media, covarianza, escala y rotacion (basado en VIGS-Fusion)
        // ==========================================================================
        // Variables con nombres coherentes con thesis.tex
        Eigen::Map<const Eigen::Vector3f> mu_w((float *)&positions[idx]);
        Eigen::Map<const Eigen::Vector3f> c_w((float *)&camera_pose.position);
        Eigen::Map<const Eigen::Quaternionf> q_cw((float *)&camera_pose.orientation);
        Eigen::Vector3f mu_c = q_cw.inverse() * (mu_w - c_w);
        Eigen::Map<const Eigen::Quaternionf> q_g((float *)&orientations[idx]);
        Eigen::Map<const Eigen::Vector3f> s((float *)&scales[idx]);

        // J = d pi / d X_c (proyeccion perspectiva), y W = R_cw (world->camera)
        Eigen::Matrix<float, 3, 3> J{{intrinsics.f.x / mu_c.z(), 0.f, -intrinsics.f.x * mu_c.x() / (mu_c.z() * mu_c.z())},
                                     {0.f, intrinsics.f.y / mu_c.z(), -intrinsics.f.y * mu_c.y() / (mu_c.z() * mu_c.z())},
                                     {0.f, 0.f, 1.f}};

        Eigen::Matrix3f R_cw = q_cw.inverse().toRotationMatrix();
        Eigen::Matrix3f R_g = q_g.toRotationMatrix();

        // dL/dmu_2d y dL/dz (vienen de optimizeGaussiansPerGaussianPass)
        Eigen::Vector3f dL_dmu_2d(delta_2d.meanImg.x, delta_2d.meanImg.y, 0.f);

        // T = J * R_cw, Sigma' = T * (R_g S^2 R_g^T) * T^T
        const Eigen::Matrix<float, 3, 3> T = J * R_cw;
        const Eigen::Matrix3f RS = R_g * s.asDiagonal();
        const Eigen::Matrix<float, 3, 3> M = T * RS;
        const Eigen::Matrix3f Sigma_prime = M * M.transpose();
        Eigen::Matrix3f Sigma_prime_inv = Sigma_prime.inverse();
        // Revisar si se puede mejorar para no invertir en cada iteracion.

        // dL/dmu_3d = R_cw^T J^T dL/dmu_2d + dL/dz * [mu_c.x/z, mu_c.y/z, 1]^T
        Eigen::Vector3f dL_dmu_3d = q_cw * (J.transpose() * dL_dmu_2d + delta_2d.depth * Eigen::Vector3f(mu_c.x() / mu_c.z(), mu_c.y() / mu_c.z(), 1.f));

        // dL/dp_hat viene del termino de distorsion: delta_2d.pHat
        Eigen::Vector2f dL_dp_hat(delta_2d.pHat.x, delta_2d.pHat.y);

        // d p_hat / d mu_c (ver deduccion en thesis.tex)
        float norm2_mu_c = mu_c.dot(mu_c);
        float norm_mu_c = sqrtf(norm2_mu_c);
        float inv_norm3 = 1.f / (norm2_mu_c * norm_mu_c);
        Eigen::Vector3f dp_hat_d_mu(-mu_c.x() * mu_c.z() * inv_norm3,
                                    -mu_c.y() * mu_c.z() * inv_norm3,
                                    (mu_c.x() * mu_c.x() + mu_c.y() * mu_c.y()) * inv_norm3);
        dL_dmu_3d += (q_cw * (dp_hat_d_mu * ((Sigma_prime_inv(2, 0) * dL_dp_hat.x() + Sigma_prime_inv(2, 1) * dL_dp_hat.y()) / Sigma_prime_inv(2, 2))));

        // dL/dSigma' a partir de dL/dSigma'^{-1} (conica 2D)
        float sigma_xx = Sigma_prime(0, 0) + 0.001f;
        float sigma_xy = Sigma_prime(1, 0);
        float sigma_yy = Sigma_prime(1, 1) + 0.001f;

        float denom = sigma_xx * sigma_yy - sigma_xy * sigma_xy;
        float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

        float3 dL_dSigmaInv2d = delta_2d.invSigmaImg;

        float dL_dSigma_xx = denom2inv * (-sigma_yy * sigma_yy * dL_dSigmaInv2d.x + 2 * sigma_xy * sigma_yy * dL_dSigmaInv2d.y + (denom - sigma_xx * sigma_yy) * dL_dSigmaInv2d.z);
        float dL_dSigma_yy = denom2inv * (-sigma_xx * sigma_xx * dL_dSigmaInv2d.z + 2 * sigma_xx * sigma_xy * dL_dSigmaInv2d.y + (denom - sigma_xx * sigma_yy) * dL_dSigmaInv2d.x);
        float dL_dSigma_xy = denom2inv * 2 * (sigma_xy * sigma_yy * dL_dSigmaInv2d.x - (denom + 2 * sigma_xy * sigma_xy) * dL_dSigmaInv2d.y + sigma_xx * sigma_xy * dL_dSigmaInv2d.z);

        const Eigen::Matrix3f Sigma_3d = RS * (RS.transpose());

        // dL/dT desde dL/dSigma' (producto T * Sigma_3d * T^T)
        float dL_dT00 = 2 * (T(0, 0) * Sigma_3d(0, 0) + T(0, 1) * Sigma_3d(0, 1) + T(0, 2) * Sigma_3d(0, 2)) * dL_dSigma_xx +
                        (T(1, 0) * Sigma_3d(0, 0) + T(1, 1) * Sigma_3d(0, 1) + T(1, 2) * Sigma_3d(0, 2)) * dL_dSigma_xy;
        float dL_dT01 = 2 * (T(0, 0) * Sigma_3d(1, 0) + T(0, 1) * Sigma_3d(1, 1) + T(0, 2) * Sigma_3d(1, 2)) * dL_dSigma_xx +
                        (T(1, 0) * Sigma_3d(1, 0) + T(1, 1) * Sigma_3d(1, 1) + T(1, 2) * Sigma_3d(1, 2)) * dL_dSigma_xy;
        float dL_dT02 = 2 * (T(0, 0) * Sigma_3d(2, 0) + T(0, 1) * Sigma_3d(2, 1) + T(0, 2) * Sigma_3d(2, 2)) * dL_dSigma_xx +
                        (T(1, 0) * Sigma_3d(2, 0) + T(1, 1) * Sigma_3d(2, 1) + T(1, 2) * Sigma_3d(2, 2)) * dL_dSigma_xy;
        float dL_dT10 = 2 * (T(1, 0) * Sigma_3d(0, 0) + T(1, 1) * Sigma_3d(0, 1) + T(1, 2) * Sigma_3d(0, 2)) * dL_dSigma_yy +
                        (T(0, 0) * Sigma_3d(0, 0) + T(0, 1) * Sigma_3d(0, 1) + T(0, 2) * Sigma_3d(0, 2)) * dL_dSigma_xy;
        float dL_dT11 = 2 * (T(1, 0) * Sigma_3d(1, 0) + T(1, 1) * Sigma_3d(1, 1) + T(1, 2) * Sigma_3d(1, 2)) * dL_dSigma_yy +
                        (T(0, 0) * Sigma_3d(1, 0) + T(0, 1) * Sigma_3d(1, 1) + T(0, 2) * Sigma_3d(1, 2)) * dL_dSigma_xy;
        float dL_dT12 = 2 * (T(1, 0) * Sigma_3d(2, 0) + T(1, 1) * Sigma_3d(2, 1) + T(1, 2) * Sigma_3d(2, 2)) * dL_dSigma_yy +
                        (T(0, 0) * Sigma_3d(2, 0) + T(0, 1) * Sigma_3d(2, 1) + T(0, 2) * Sigma_3d(2, 2)) * dL_dSigma_xy;

        // Aporte de p_hat a dL/dSigma'^{-1} y dL/dSigma'
        float z_over_t = mu_c.z() / norm_mu_c;
        Eigen::Matrix3f dL_dSigmaInv;
        dL_dSigmaInv << 0.f, 0.f, 0.5f * z_over_t * dL_dp_hat.x() / Sigma_prime_inv(2, 2),
            0.f, 0.f, 0.5f * z_over_t * dL_dp_hat.y() / Sigma_prime_inv(2, 2),
            0.5f * z_over_t * dL_dp_hat.x() / Sigma_prime_inv(2, 2), 0.5f * z_over_t * dL_dp_hat.y() / Sigma_prime_inv(2, 2), -z_over_t * (dL_dp_hat.x() * Sigma_prime_inv(2, 0) + dL_dp_hat.y() * Sigma_prime_inv(2, 1)) / (Sigma_prime_inv(2, 2) * Sigma_prime_inv(2, 2));

        Eigen::Matrix3f dL_dSigma_prime = -Sigma_prime_inv * dL_dSigmaInv * Sigma_prime_inv;

        Eigen::Matrix3f TSigma3D = T * Sigma_3d;
        Eigen::Matrix3f Sigma3DT = Sigma_3d * T.transpose();
        dL_dT00 += TSigma3D.col(0).dot(dL_dSigma_prime.col(0)) + Sigma3DT.row(0).dot(dL_dSigma_prime.row(0));
        dL_dT01 += TSigma3D.col(1).dot(dL_dSigma_prime.col(0)) + Sigma3DT.row(1).dot(dL_dSigma_prime.row(0));
        dL_dT02 += TSigma3D.col(2).dot(dL_dSigma_prime.col(0)) + Sigma3DT.row(2).dot(dL_dSigma_prime.row(0));
        dL_dT10 += TSigma3D.col(0).dot(dL_dSigma_prime.col(1)) + Sigma3DT.row(0).dot(dL_dSigma_prime.row(1));
        dL_dT11 += TSigma3D.col(1).dot(dL_dSigma_prime.col(1)) + Sigma3DT.row(1).dot(dL_dSigma_prime.row(1));
        dL_dT12 += TSigma3D.col(2).dot(dL_dSigma_prime.col(1)) + Sigma3DT.row(2).dot(dL_dSigma_prime.row(1));

        // dL/dJ y dL/dmu_c (via dJ/dmu_c)
        float dL_dJ00 = R_cw(0, 0) * dL_dT00 + R_cw(0, 1) * dL_dT01 + R_cw(0, 2) * dL_dT02;
        float dL_dJ02 = R_cw(2, 0) * dL_dT00 + R_cw(2, 1) * dL_dT01 + R_cw(2, 2) * dL_dT02;
        float dL_dJ11 = R_cw(1, 0) * dL_dT10 + R_cw(1, 1) * dL_dT11 + R_cw(1, 2) * dL_dT12;
        float dL_dJ12 = R_cw(2, 0) * dL_dT10 + R_cw(2, 1) * dL_dT11 + R_cw(2, 2) * dL_dT12;

        float inv_z = 1.f / mu_c.z();
        float inv_z2 = inv_z * inv_z;
        float inv_z3 = inv_z2 * inv_z;

        float dL_dmu_cx = -intrinsics.f.x * inv_z2 * dL_dJ02;
        float dL_dmu_cy = -intrinsics.f.y * inv_z2 * dL_dJ12;
        float dL_dmu_cz = -intrinsics.f.x * inv_z2 * dL_dJ00 - intrinsics.f.y * inv_z2 * dL_dJ11 + (2 * intrinsics.f.x * mu_c.x()) * inv_z3 * dL_dJ02 + (2 * intrinsics.f.y * mu_c.y()) * inv_z3 * dL_dJ12;

        dL_dmu_3d += q_cw * Eigen::Vector3f(dL_dmu_cx, dL_dmu_cy, dL_dmu_cz);

        // Guardamos dL/dmu_3d (posicion)
        delta_3d.position.x = eta_n * dL_dmu_3d.x();
        delta_3d.position.y = eta_n * dL_dmu_3d.y();
        delta_3d.position.z = eta_n * dL_dmu_3d.z();

        // dL/dSigma_3d = T^T * dL/dSigma' * T
        Eigen::Matrix3f dL_dSigma_3d;
        dL_dSigma_3d(0, 0) = (T(0, 0) * T(0, 0) * dL_dSigma_xx + T(0, 0) * T(1, 0) * dL_dSigma_xy + T(1, 0) * T(1, 0) * dL_dSigma_yy);
        dL_dSigma_3d(1, 1) = (T(0, 1) * T(0, 1) * dL_dSigma_xx + T(0, 1) * T(1, 1) * dL_dSigma_xy + T(1, 1) * T(1, 1) * dL_dSigma_yy);
        dL_dSigma_3d(2, 2) = (T(0, 2) * T(0, 2) * dL_dSigma_xx + T(0, 2) * T(1, 2) * dL_dSigma_xy + T(1, 2) * T(1, 2) * dL_dSigma_yy);
        dL_dSigma_3d(1, 0) = dL_dSigma_3d(0, 1) = T(0, 0) * T(0, 1) * dL_dSigma_xx + 0.5f * (T(0, 0) * T(1, 1) + T(0, 1) * T(1, 0)) * dL_dSigma_xy + T(1, 0) * T(1, 1) * dL_dSigma_yy;
        dL_dSigma_3d(2, 0) = dL_dSigma_3d(0, 2) = T(0, 0) * T(0, 2) * dL_dSigma_xx + 0.5f * (T(0, 0) * T(1, 2) + T(0, 2) * T(1, 0)) * dL_dSigma_xy + T(1, 0) * T(1, 2) * dL_dSigma_yy;
        dL_dSigma_3d(2, 1) = dL_dSigma_3d(1, 2) = T(0, 2) * T(0, 1) * dL_dSigma_xx + 0.5f * (T(0, 1) * T(1, 2) + T(0, 2) * T(1, 1)) * dL_dSigma_xy + T(1, 1) * T(1, 2) * dL_dSigma_yy;

        dL_dSigma_3d += T.transpose() * dL_dSigma_prime * T;

        // dL/dS y dL/dR con Sigma_3d = R S^2 R^T
        Eigen::Matrix3f dL_dM = 2.0f * RS.transpose() * dL_dSigma_3d;
        Eigen::Vector3f dL_ds = Eigen::Vector3f(R_g(0, 0) * dL_dM(0, 0) + R_g(1, 0) * dL_dM(0, 1) + R_g(2, 0) * dL_dM(0, 2),
                                                R_g(0, 1) * dL_dM(1, 0) + R_g(1, 1) * dL_dM(1, 1) + R_g(2, 1) * dL_dM(1, 2),
                                                R_g(0, 2) * dL_dM(2, 0) + R_g(1, 2) * dL_dM(2, 1) + R_g(2, 2) * dL_dM(2, 2));

        const float3 s_val = scales[idx];

        // Regularizacion isotropica: dL_iso/ds
        float mean_s = (s_val.x + s_val.y + s_val.z) / 3.f;
        float3 dl_iso = make_float3(s_val.x - mean_s,
                        s_val.y - mean_s,
                        s_val.z - mean_s);
        dL_ds -= lambda_iso * (1.f / 3.f) * Eigen::Vector3f(2.f * dl_iso.x - dl_iso.y - dl_iso.z, -dl_iso.x + 2.f * dl_iso.y - dl_iso.z, -dl_iso.x - dl_iso.y + 2.f * dl_iso.z);

        // dL/dR en espacio tangente: -sum_i R_i x dL/dM_i
        dL_dM.row(0) *= s_val.x;
        dL_dM.row(1) *= s_val.y;
        dL_dM.row(2) *= s_val.z;

        Eigen::Vector3f dL_dtheta = -eta_n * (R_g.row(0).cross(dL_dM.col(0)) + R_g.row(1).cross(dL_dM.col(1)) + R_g.row(2).cross(dL_dM.col(2)));

        delta_3d.orientation.x = dL_dtheta.x();
        delta_3d.orientation.y = dL_dtheta.y();
        delta_3d.orientation.z = dL_dtheta.z();

        delta_3d.scale.x = eta_n * dL_ds.x();
        delta_3d.scale.y = eta_n * dL_ds.y();
        delta_3d.scale.z = eta_n * dL_ds.z();

        delta_gaussians_3d[idx] = delta_3d;
    }

    __inline__ __device__ float adamStep(float &m,
                                         float &v,
                                         float grad,
                                         const float eta,
                                         const float alpha1,
                                         const float beta1,
                                         const float beta1t,
                                         const float alpha2,
                                         const float beta2,
                                         const float beta2t,
                                         const float epsilon)
    {
        // Actualizamos momento y varianza
        m = alpha1 * grad + beta1 * m;
        v = alpha2 * grad * grad + beta2 * v;
        
        // Aplicamos la correccion
        float m_hat = beta1t * m;
        float v_hat = beta2t * v;
        
        // Devolvemos el siguiente paso
        return eta * m_hat * __frsqrt_rn(v_hat + epsilon);
    }

    __inline__ __device__ float3 adamStep(float3 &m,
                                          float3 &v,
                                          float3 grad,
                                          const float eta,
                                          const float alpha1,
                                          const float beta1,
                                          const float beta1t,
                                          const float alpha2,
                                          const float beta2,
                                          const float beta2t,
                                          const float epsilon)
    {
        float3 res;
        
        res.x = adamStep(m.x, v.x, grad.x,
                        eta,
                        alpha1, beta1, beta1t,
                        alpha2, beta2, beta2t,
                        epsilon);
        res.y = adamStep(m.y, v.y, grad.y,
                        eta,
                        alpha1, beta1, beta1t,
                        alpha2, beta2, beta2t,
                        epsilon);
        res.z = adamStep(m.z, v.z, grad.z,
                        eta,
                        alpha1, beta1, beta1t,
                        alpha2, beta2, beta2t,
                        epsilon);
        
        return res;
    }

    __global__ void updateGaussiansParametersAdam_kernel(
        float3 *positions,
        float3 *scales,
        float4 *orientations,
        float3 *colors,
        float *alphas,
        AdamStateGaussian3D *adam_states,
        const DeltaGaussian3D *deltas_3d,
        float adam_eta,
        float adam_beta1,
        float adam_beta2,
        float adam_eps,
        int n_gaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_gaussians) {
            return;
        }

        DeltaGaussian3D delta = deltas_3d[idx];
        if (delta.n == 0) {
            return;  // Skip Gaussians with no gradient
        }
        
        AdamStateGaussian3D adam_state = adam_states[idx];

        // Avanzamos un paso
        adam_state.k += 1.f;

        // Hiperparametro alpha = 1 - beta
        float alpha1 = 1.f - adam_beta1;
        float alpha2 = 1.f - adam_beta2;

        // Correccion de bias: 1 / (1 - beta^t)
        float beta1t = __frcp_rn(1.f - __powf(adam_beta1, adam_state.k));
        float beta2t = __frcp_rn(1.f - __powf(adam_beta2, adam_state.k));

        // Cargamos parametros
        float3 position = positions[idx];
        float3 scale = scales[idx];
        float4 orientation = orientations[idx];
        float3 color = colors[idx];
        float alpha = alphas[idx];

        // La posicion se actualiza directamente
        position += adamStep(adam_state.m_position,
                            adam_state.v_position,
                            delta.position,
                            adam_eta,
                            alpha1, adam_beta1, beta1t,
                            alpha2, adam_beta2, beta2t,
                            adam_eps);

        // La orientacion se actualiza en el espacio tangente de SO(3) para mantener la normalizacion del cuaternion
        float3 dq = adamStep(adam_state.m_orientation,
                            adam_state.v_orientation,
                            delta.orientation,
                            adam_eta,
                            alpha1, adam_beta1, beta1t,
                            alpha2, adam_beta2, beta2t,
                            adam_eps);
        Eigen::Map<Eigen::Quaternionf> q_gauss((float *)&orientation);
        q_gauss = q_gauss * Eigen::Quaternionf(1.f, 0.5f * dq.x, 0.5f * dq.y, 0.5f * dq.z);
        q_gauss.normalize();

        // La escala se actualiza directamente
        scale += adamStep(adam_state.m_scale,
                         adam_state.v_scale,
                         delta.scale,
                         adam_eta,
                         alpha1, adam_beta1, beta1t,
                         alpha2, adam_beta2, beta2t,
                         adam_eps);

        // El color se actualiza directamente
        color += adamStep(adam_state.m_color,
                         adam_state.v_color,
                         delta.color,
                         adam_eta,
                         alpha1, adam_beta1, beta1t,
                         alpha2, adam_beta2, beta2t,
                         adam_eps);

        // La opacidad se actualiza en el espacio logit para mantenerla en el rango (0, 1)
        float dalpha = adamStep(adam_state.m_alpha,
                               adam_state.v_alpha,
                               delta.alpha,
                               adam_eta,
                               alpha1, adam_beta1, beta1t,
                               alpha2, adam_beta2, beta2t,
                               adam_eps);
        float alpha_s = __logf(alpha / (1.f - alpha)) + dalpha / (alpha - alpha * alpha);
        alpha = max(0.01f, min(0.99f, 1.f / (1.f + __expf(-alpha_s))));

        //Actualizamos los parametros y clampeamos para mantener valores utiles e interpretables
        positions[idx] = position;
        scales[idx] = max(make_float3(0.001f), scale);
        orientations[idx] = orientation;
        colors[idx] = min(make_float3(1.f, 1.f, 1.f), max(make_float3(0.f, 0.f, 0.f), color));
        alphas[idx] = alpha;


        adam_states[idx] = adam_state;
    }

}