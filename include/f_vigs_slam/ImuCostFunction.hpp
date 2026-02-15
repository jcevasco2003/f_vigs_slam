#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <memory>

#include "f_vigs_slam/Preintegration.hpp"

namespace f_vigs_slam
{
    class ImuCostFunction : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
    {
    public:
        explicit ImuCostFunction(std::shared_ptr<Preintegration> preintegration);
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    private:
        std::shared_ptr<Preintegration> pre_integration_;
        const Eigen::Vector3d G_{0.0, 0.0, 9.8};
    };
}