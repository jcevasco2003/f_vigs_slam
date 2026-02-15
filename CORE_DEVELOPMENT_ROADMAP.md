# Core Development Roadmap

## Overview
Estructura base para implementar el núcleo (core) del SLAM con Gaussian Splatting. El pipeline principal está en `GSSlam::compute()`.

## Arquitectura

```
compute(rgb, depth, odometry_pose)
    ├── initializeFirstFrame()     [GPU init + state setup]
    ├── rasterize()                [Render gaussians to GPU]
    ├── rasterizeFill()            [Fill rendered images]
    ├── computeRenderingErrors()   [Compute error maps]
    ├── optimizePose()             [Optimize camera pose]
    ├── optimizeGaussians()        [Optimize gaussian params]
    ├── densify()                  [Split high-variance gaussians]
    ├── prune()                    [Remove low-opacity gaussians]
    └── addKeyframe()              [Store current frame as keyframe]
```

## Implementation Phases

### Phase 1: Renderization (PRIORITY: HIGH)
**Goal:** Render gaussians to 2D images for visual comparison

**Files:**
- `src/GSCudaKernels.cu` - Add rendering kernels
- `src/GSSlam.cu` - Implement `rasterize()` and `rasterizeFill()`

**Tasks:**
- [ ] Create tile-based rasterization kernel
- [ ] Implement gaussian splatting forward pass
- [ ] Compute rendered RGB and depth images
- [ ] Add normal rendering for debugging

**Dependencies:**
- Thrust algorithms (sort by depth)
- Atomic operations for tile building
- Covariance matrix handling

---

### Phase 2: Error Computation (PRIORITY: HIGH)
**Goal:** Measure difference between rendered and ground-truth images

**Files:**
- `src/GSCudaKernels.cu` - Add error computation kernels
- `src/GSSlam.cu` - Implement `computeRenderingErrors()`

**Tasks:**
- [ ] Compute per-pixel RGB error (L1 or L2)
- [ ] Compute per-pixel depth error
- [ ] Weighted error map (w_depth * depth_error + rgb_error)
- [ ] Gradient computation for optimization

**Dependencies:**
- Error metrics (MSE, L1, etc.)
- Weighting parameters (w_depth, w_dist)

---

### Phase 3: Pose Optimization (PRIORITY: HIGH)
**Goal:** Refine camera pose using visual error

**Files:**
- `src/GSSlam.cu` - Implement `optimizePose()`
- New: `src/PoseCostFunction.cu` (optional)

**Tasks:**
- [ ] Compute jacobian of rendering w.r.t pose (6D: 3 translation + 3 rotation)
- [ ] Implement Gauss-Newton or gradient descent
- [ ] Handle pose updates (SE3 or local parameterization)
- [ ] Multi-level optimization (coarse-to-fine)

**Key Parameters:**
- `pose_iterations_` (default: 4)
- `eta_pose_` (default: 0.01)

---

### Phase 4: Gaussian Optimization (PRIORITY: HIGH)
**Goal:** Refine gaussian parameters (position, scale, color, opacity)

**Files:**
- `src/GSSlam.cu` - Implement `optimizeGaussians()`
- `src/GSCudaKernels.cu` - Add gradient computation kernels

**Tasks:**
- [ ] Compute error gradients w.r.t each gaussian parameter
- [ ] Implement Adam optimizer for gaussian updates
- [ ] Update position, scale, orientation, color, opacity
- [ ] Clamp values to valid ranges

**Key Parameters:**
- `gaussian_iterations_` (default: 10)
- `eta_gaussian_` (default: 0.002)

---

### Phase 5: Densification & Pruning (PRIORITY: MEDIUM)
**Goal:** Improve gaussian representation quality

**Files:**
- `src/GSSlam.cu` - Implement `densify()` and `prune()`
- `src/GSCudaKernels.cu` - Add densify/prune kernels

**Tasks - Densify:**
- [ ] Identify high-error gaussians
- [ ] Split gaussians (create 2 smaller gaussians from 1)
- [ ] Implement Beta-Binomial splitting strategy
- [ ] Maintain Gaussians count limits

**Tasks - Prune:**
- [ ] Remove gaussians with low opacity
- [ ] Remove duplicates/overlaps
- [ ] Memory management

---

### Phase 6: Keyframe Management (PRIORITY: MEDIUM)
**Goal:** Store and manage important frames for later optimization

**Files:**
- `src/GSSlam.cu` - Implement `addKeyframe()`
- New: `include/f_vigs_slam/KeyframeSelector.hpp` (already exists)

**Tasks:**
- [ ] Store keyframe data (pose, images, gaussians count)
- [ ] Keyframe selection criteria (baseline, uncertainty, etc.)
- [ ] Multi-view optimization placeholder

---

### Phase 7: Advanced Features (PRIORITY: LOW)
**Goal:** Enhance robustness and accuracy

**Tasks:**
- [ ] Loop closure detection
- [ ] Global optimization (bundle adjustment)
- [ ] IMU preintegration fusion
- [ ] Marginalization for memory efficiency
- [ ] Multi-resolution rendering
- [ ] Viewer/debugging visualization

---

## Data Structures Ready

```cpp
// In GSSlam.cuh
cv::cuda::GpuMat rgb_gpu_;           // Input RGB
cv::cuda::GpuMat depth_gpu_;         // Input depth
cv::cuda::GpuMat rendered_rgb_gpu_;  // Rendered output
cv::cuda::GpuMat rendered_depth_gpu_;// Rendered output
cv::cuda::GpuMat error_map_gpu_;     // Error map

thrust::device_vector<float3> gaussian_gradients_;
thrust::device_vector<float> opacity_gradients_;

// State arrays
double P_cur_[7];   // Current pose
double P_prev_[7];  // Previous pose
double VB_cur_[9];  // Velocity + bias
double VB_prev_[9];
```

---

## Configuration (ROS2 Parameters)

```yaml
gs_slam_node:
  ros__parameters:
    # Optimization parameters
    pose_iterations: 4          # Iterations for pose optimization
    gaussian_iterations: 10     # Iterations for gaussian optimization
    eta_pose: 0.01             # Learning rate for pose
    eta_gaussian: 0.002        # Learning rate for gaussians
    
    # Initialization
    gauss_init_size_px: 7      # Sample grid size (pixels)
    gauss_init_scale: 0.01     # Initial gaussian scale
    
    # Error weighting
    w_depth: 1.0               # Depth error weight
    w_dist: 0.1                # Distance consistency weight
```

---

## Testing Strategy

1. **Unit Tests:**
   - [ ] Render consistency (same pose = same output)
   - [ ] Error computation correctness
   - [ ] Gradient computation (numerical vs analytical)

2. **Integration Tests:**
   - [ ] Full compute() pipeline
   - [ ] Convergence under noise
   - [ ] Memory usage tracking

3. **Validation:**
   - [ ] Compare with VIGS-Fusion outputs
   - [ ] Benchmark performance (FPS, accuracy)
   - [ ] Runtime profiling

---

## Performance Targets

- **FPS:** 10-30 Hz (depends on resolution and gaussian count)
- **Memory:** < 4GB GPU VRAM for 50k gaussians
- **Latency:** < 100ms per frame
- **Accuracy:** < 5cm trajectory error on standard benchmarks

---

## Next Steps

1. Start with **Phase 1 (Renderization)** - fundamental for everything else
2. Reference `GaussianSplattingKernels.cu` in VIGS-Fusion for implementation details
3. Test incrementally with debug visualizations
4. Profile GPU memory and compute time regularly
