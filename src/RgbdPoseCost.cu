#include "f_vigs_slam/RgbdPoseCost.hpp"
#include "f_vigs_slam/GSSlam.cuh"
#include <iostream>

namespace f_vigs_slam
{
    RgbdPoseCostFunction::RgbdPoseCostFunction(GSSlam* gs_slam)
        : gs_slam_(gs_slam), level_(0)
    {
    }

    RgbdPoseCostFunction::~RgbdPoseCostFunction()
    {
    }

    void RgbdPoseCostFunction::update(int level)
    {
        level_ = level;
    }

    bool RgbdPoseCostFunction::Evaluate(double const* const* parameters,
                                      double* residuals,
                                      double** jacobians) const
    {
        return const_cast<RgbdPoseCostFunction*>(this)->EvaluateNonConst(
            parameters, residuals, jacobians);
    }

    bool RgbdPoseCostFunction::EvaluateNonConst(double const* const* parameters,
                                              double* residuals,
                                              double** jacobians)
    {
        // ============================================================
        // PASO 1: Extraer pose IMU de parámetros Ceres
        // ============================================================
        Eigen::Map<const Eigen::Vector3d> P_imu(parameters[0]);
        Eigen::Map<const Eigen::Quaterniond> Q_imu(parameters[0] + 3);

        // ============================================================
        // PASO 2: Calcular JtJ y Jtr mediante kernel GPU
        // ============================================================
        // Este método hace:
        // - Convertir pose IMU → pose cámara (transformación extrínseca)
        // - Renderizar gaussianas desde pose cámara
        // - Calcular residuales RGB-D pixel a pixel
        // - Acumular Hessiana aproximada J^T * J y gradiente J^T * r
        Eigen::Matrix<double, 6, 6> JtJ;
        Eigen::Vector<double, 6> Jtr;
        
        gs_slam_->computeRgbdPoseJacobians(JtJ, Jtr, level_, P_imu, Q_imu);

        // ============================================================
        // PASO 3: Regularizar JtJ mediante eigenvalue decomposition
        // ============================================================
        // Esto previene matrices singulares cuando hay poca información visual
        // Similar a añadir regularización de Tikhonov pero adaptativa
        double eps = 1e-4;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(JtJ);
        
        // Filtrar eigenvalores muy pequeños (ruido numérico)
        Eigen::VectorXd S = Eigen::VectorXd(
            (es.eigenvalues().array() > eps).select(es.eigenvalues().array(), 0));
        Eigen::VectorXd S_inv = Eigen::VectorXd(
            (es.eigenvalues().array() > eps).select(es.eigenvalues().array().inverse(), 0));
        
        // Raíces cuadradas para pseudo-jacobiano
        Eigen::VectorXd S_sqrt = S.cwiseSqrt();
        Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
        
        // ============================================================
        // PASO 4: Calcular pseudo-residuales y pseudo-jacobiano
        // ============================================================
        // En lugar de pasar J y r directamente, Ceres recibe:
        // J* = sqrt(Λ) * U^T  donde JtJ = U * Λ * U^T
        // r* = sqrt(Λ^-1) * U^T * Jtr
        // Esto garantiza que J*^T * J* = JtJ exactamente
        Eigen::MatrixXd J_star = S_sqrt.asDiagonal() * es.eigenvectors().transpose();
        Eigen::VectorXd r_star = S_inv_sqrt.asDiagonal() * es.eigenvectors().transpose() * Jtr;

        // ============================================================
        // PASO 5: Copiar residuales a salida de Ceres
        // ============================================================
        residuals[0] = r_star(0);
        residuals[1] = r_star(1);
        residuals[2] = r_star(2);
        residuals[3] = r_star(3);
        residuals[4] = r_star(4);
        residuals[5] = r_star(5);

        // ============================================================
        // PASO 6: Calcular jacobiano si es requerido
        // ============================================================
        if (jacobians != nullptr && jacobians[0] != nullptr)
        {
            // Jacobiano es 6×7 (6 residuales, 7 parámetros de pose)
            // Última columna es cero porque quaternion está sobreparametrizado
            // (Ceres maneja esto con PoseLocalParameterization)
            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_pose(jacobians[0]);
            J_pose.setZero();
            
            // Solo las primeras 6 columnas tienen información (espacio tangente)
            J_pose.block<6, 6>(0, 0) = J_star;
            
            // Nota: La transformación de espacio cámara a espacio IMU
            // ya está incluida en computeRgbdPoseJacobians()
        }

        return true;
    }

} // namespace f_vigs_slam
