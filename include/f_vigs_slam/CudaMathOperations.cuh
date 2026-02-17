#pragma once

#include <cuda_runtime.h>

// ============================================================================
// Float3 Operators
// ============================================================================

__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3 &a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator*(float s, const float3 &a)
{
    return a * s;
}

__host__ __device__ inline float3& operator+=(float3 &a, const float3 &b)
{
    a = a + b;
    return a;
}

__host__ __device__ inline float3 cross(const float3 &a, const float3 &b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

// ============================================================================
// Min/Max for float3
// ============================================================================

__host__ __device__ inline float3 max(const float3 &a, const float3 &b)
{
    return make_float3(
        a.x > b.x ? a.x : b.x,
        a.y > b.y ? a.y : b.y,
        a.z > b.z ? a.z : b.z);
}

__host__ __device__ inline float3 min(const float3 &a, const float3 &b)
{
    return make_float3(
        a.x < b.x ? a.x : b.x,
        a.y < b.y ? a.y : b.y,
        a.z < b.z ? a.z : b.z);
}
