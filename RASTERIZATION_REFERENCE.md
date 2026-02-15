# Rasterization Reference Guide

## Estructura General del Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  prepareRasterization()                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. computeScreenSpaceParams_kernel                  │   │
│  │    - Transform 3D → Camera → 2D                     │   │
│  │    - Compute covariance 2D                          │   │
│  │    - Generate hashes per tile                       │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 2. thrust::sort_by_key(hashes, indices)            │   │
│  │    - Sort by depth within each tile                 │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 3. computeTileRanges_kernel                         │   │
│  │    - Find [start, end] range per tile               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  rasterize()                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 4. rasterizeGaussians_kernel                        │   │
│  │    - Tile-based rendering                           │   │
│  │    - Alpha blending per pixel                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Fórmulas Matemáticas

### 1. Transformación 3D → Cámara

**Quaternion → Rotation Matrix:**
```
q = (w, x, y, z)

R = | 1-2(y²+z²)   2(xy-wz)     2(xz+wy)   |
    | 2(xy+wz)     1-2(x²+z²)   2(yz-wx)   |
    | 2(xz-wy)     2(yz+wx)     1-2(x²+y²) |
```

**World → Camera:**
```
P_cam = R_cam * (P_world - T_cam)

donde:
  R_cam = rotation matrix from camera_pose.orientation
  T_cam = camera_pose.position
```

---

### 2. Proyección 2D

```
u = fx * (P_cam.x / P_cam.z) + cx
v = fy * (P_cam.y / P_cam.z) + cy
depth = P_cam.z
```

**Verificación de frustum:**
```
if (depth <= 0 || u < 0 || u >= width || v < 0 || v >= height):
    descartar gaussiana
```

---

### 3. Covarianza 3D → 2D

**Paso 1: Construir covarianza 3D de la gaussiana**
```
S = diag(scale.x, scale.y, scale.z)
R_gauss = quaternion_to_matrix(orientation)
Σ_world = R_gauss * S * S^T * R_gauss^T
```

**Paso 2: Transformar a cámara**
```
Σ_cam = R_cam * Σ_world * R_cam^T
```

**Paso 3: Jacobiano de proyección**
```
fx, fy = focal lengths
x, y, z = P_cam.x, P_cam.y, P_cam.z

J = | fx/z    0      -fx*x/z² |
    |   0    fy/z    -fy*y/z² |
```

**Paso 4: Covarianza 2D**
```
Σ' = J * Σ_cam[0:2, 0:2] * J^T

Resultado: matriz 2x2
Σ' = | σ_xx  σ_xy |
     | σ_xy  σ_yy |
```

**Paso 5: Invertir covarianza**
```
det = σ_xx * σ_yy - σ_xy²

Σ_inv = (1/det) * |  σ_yy  -σ_xy |
                   | -σ_xy   σ_xx |
```

Guardar como `float3(Σ_inv[0,0], Σ_inv[0,1], Σ_inv[1,1])`

---

### 4. Radio de Cobertura y Tiles

**Radio de influencia (3-sigma):**
```
eigenvalues = solve(det(Σ' - λI) = 0)
max_eigenvalue = max(eigenvalues)
radius = 3 * sqrt(max_eigenvalue)
```

**Tiles cubiertos:**
```
tile_min_x = floor((u - radius) / tile_size.x)
tile_max_x = ceil((u + radius) / tile_size.x)
tile_min_y = floor((v - radius) / tile_size.y)
tile_max_y = ceil((v + radius) / tile_size.y)

Clamp to [0, num_tiles-1]
```

---

### 5. Hash para Sorting

```
tileID = tile_y * num_tiles.x + tile_x
depth_uint = *(uint32_t*)&depth  // reinterpret float as uint

hash = ((uint64_t)tileID << 32) | depth_uint
```

**Propiedad:** Al ordenar por hash:
- Primero se agrupan por tile (bits altos)
- Dentro del tile, se ordenan por profundidad (bits bajos)

---

### 6. Alpha Blending (Rasterización)

**Para cada pixel (px, py):**

```
pixel_pos = (px + 0.5, py + 0.5)

C_accumulated = (0, 0, 0)
D_accumulated = 0
T = 1.0  // Transmittance

for each gaussian i in tile:
    delta = pixel_pos - gaussian_center - p_hat
    
    // Evaluar gaussiana 2D
    power = -0.5 * (delta.x² * σ_xx + 2*delta.x*delta.y * σ_xy + delta.y² * σ_yy)
    
    if power < -10:
        continue
    
    α = opacity * exp(power)
    
    if α < 0.01:
        continue
    
    // Alpha blending
    C_accumulated += T * α * color_i
    D_accumulated += T * α * depth_i
    T *= (1 - α)
    
    if T < 0.001:
        break  // Early stopping

// Agregar background
C_accumulated += T * bg_color

output_color[px, py] = C_accumulated
output_depth[px, py] = D_accumulated
```

---

## Helpers CUDA Útiles

### Quaternion → Matrix
```cuda
__device__ inline void quaternionToMatrix(const float4& q, float R[3][3]) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    
    R[0][0] = 1 - 2*(y*y + z*z);
    R[0][1] = 2*(x*y - w*z);
    R[0][2] = 2*(x*z + w*y);
    
    R[1][0] = 2*(x*y + w*z);
    R[1][1] = 1 - 2*(x*x + z*z);
    R[1][2] = 2*(y*z - w*x);
    
    R[2][0] = 2*(x*z - w*y);
    R[2][1] = 2*(y*z + w*x);
    R[2][2] = 1 - 2*(x*x + y*y);
}
```

### Matrix × Vector
```cuda
__device__ inline float3 matrixVectorMul(float R[3][3], float3 v) {
    return make_float3(
        R[0][0]*v.x + R[0][1]*v.y + R[0][2]*v.z,
        R[1][0]*v.x + R[1][1]*v.y + R[1][2]*v.z,
        R[2][0]*v.x + R[2][1]*v.y + R[2][2]*v.z
    );
}
```

### Matrix 2x2 Inverse
```cuda
__device__ inline void invert2x2(float a, float b, float c, 
                                 float& inv_a, float& inv_b, float& inv_c) {
    // Input: | a  b |
    //        | b  c |
    float det = a * c - b * b;
    if (fabsf(det) < 1e-10f) {
        // Singular matrix, usar identidad
        inv_a = 1.0f; inv_b = 0.0f; inv_c = 1.0f;
        return;
    }
    float inv_det = 1.0f / det;
    inv_a = c * inv_det;
    inv_b = -b * inv_det;
    inv_c = a * inv_det;
}
```

### Float → Sortable Uint (para depth sorting)
```cuda
__device__ inline uint32_t floatToSortableUint(float f) {
    uint32_t u = *(uint32_t*)&f;
    // Si es negativo, invertir todos los bits
    // Si es positivo, invertir solo el bit de signo
    return (u & 0x80000000) ? (~u) : (u | 0x80000000);
}
```

---

## Optimizaciones Importantes

### 1. Early Stopping
- Si `T < 0.001`, no hay transmitencia → parar iteración
- Si `power < -10`, gaussiana negligible → skip
- Si `α < 0.01`, contribución despreciable → skip

### 2. Shared Memory (futuro)
- Cargar tile_ranges en shared memory
- Cargar gaussianas del tile en shared memory
- Reduce accesos globales

### 3. Warp Divergence
- Todas las gaussianas en un tile se procesan igual
- Threads en el mismo warp procesan pixels contiguos

### 4. Memory Coalescing
- Acceso a `img_positions[idx]` es coalesced si threads consecutivos acceden indices consecutivos
- Esto se garantiza con el sorting por tile

---

## Debugging Tips

### Visualizar Screen-Space
```cpp
// Después de computeScreenSpaceParams_kernel
thrust::host_vector<float3> h_positions = img_positions_;
for (int i = 0; i < 10; i++) {
    printf("Gauss %d: u=%.1f v=%.1f depth=%.2f\n", 
           i, h_positions[i].x, h_positions[i].y, h_positions[i].z);
}
```

### Verificar Sorting
```cpp
// Después de thrust::sort_by_key
thrust::host_vector<uint64_t> h_hashes = hashes_;
for (int i = 0; i < 20; i++) {
    uint32_t tileID = h_hashes[i] >> 32;
    printf("Hash %d: tileID=%u\n", i, tileID);
}
// Debe estar ordenado por tileID
```

### Visualizar Rendered Image
```cpp
cv::Mat rendered_cpu;
rendered_rgb_gpu_.download(rendered_cpu);
cv::imshow("Rendered", rendered_cpu);
cv::waitKey(0);
```

---

## Orden de Implementación Recomendado

1. **Primero:** `computeScreenSpaceParams_kernel`
   - Implementar proyección básica
   - Calcular covarianza 2D (simplificada)
   - Generar 1 hash por gaussiana (sin tile coverage)

2. **Segundo:** `computeTileRanges_kernel`
   - Binary search simple

3. **Tercero:** `rasterizeGaussians_kernel`
   - Alpha blending básico
   - Sin early stopping inicialmente

4. **Cuarto:** Refinamiento
   - Tile coverage completo (múltiples hashes por gaussiana)
   - Optimizaciones (early stopping, thresholds)

---

## Referencias

- **3D Gaussian Splatting Paper:** https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **VIGS-Fusion:** `/home/jorge/ros2_work_space/src/VIGS-Fusion/src/GaussianSplattingSlamKernels.cu`
- **Diff Gaussian Rasterization:** https://github.com/graphdeco-inria/diff-gaussian-rasterization
