#include "f_vigs_slam/PoseLocalParameterization.hpp"

namespace f_vigs_slam
{
    /**
     * @brief Helper: Convierte un vector de rotación (axis-angle) a un quaternion aproximado
     *
     * Aproximación de primer orden: exp(ω) = [1, ω/2] + error despreciable
     * El resultado no esta normalizado
     */
    inline Eigen::Quaterniond PoseLocalParameterization::deltaQ(const Eigen::Vector3d &theta)
    {
        Eigen::Quaterniond dq;
        Eigen::Vector3d half_theta = 0.5 * theta;
        dq.w() = 1.0;
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }

    bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
    {
        // Estado actual
        Eigen::Map<const Eigen::Vector3d> p(x);
        Eigen::Map<const Eigen::Quaterniond> q(x + 3);

        // Perturbación local
        Eigen::Map<const Eigen::Vector3d> dp(delta);
        Eigen::Vector3d dtheta = Eigen::Map<const Eigen::Vector3d>(delta + 3);

        // Perturbación de rotación: dq = exp(dtheta) + error despreciable
        Eigen::Quaterniond dq = deltaQ(dtheta);

        // Actualización: p_new = p + dp
        Eigen::Map<Eigen::Vector3d> p_new(x_plus_delta);
        p_new = p + dp;

        // Actualización: q_new = q (+) dq (normalizado)
        Eigen::Map<Eigen::Quaterniond> q_new(x_plus_delta + 3);
        q_new = (q * dq).normalized();

        return true;
    }

    bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
    {
        (void)x;  // Jacobiano constante para la parte lineal de posición

        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();

        // Parte de posición: dp_new/dp = I_3, dp_new/dtheta = 0
        J.block<3, 3>(0, 0).setIdentity();

        // Parte de rotación: dq_new/dp = 0, dq_new/dtheta = estructura especial
        // Para la perturbación quaternion q_new = q (+) exp(theta)
        // El jacobiano respecto a la rotación local 3D es el bloque 4x3 del producto
        // Con primer orden: dq_new/dtheta mapea a la parte vectorial
        // J.block<4, 3>(3, 3) lo maneja; se simplifica como identidad en la parte vectorial
        J.block<3, 3>(4, 3).setIdentity();

        return true;
    }

    void PoseLocalParameterization::boxMinus(const double *xi, const double *xj,
                                             double *xi_minus_xj) const
    {
        // xi: [pi, qi] con qi = [qix, qiy, qiz, qiw]
        Eigen::Map<const Eigen::Vector3d> pi(xi);
        Eigen::Map<const Eigen::Vector3d> qi_xyz(xi + 3);
        const double qi_w = xi[6];
        const Eigen::Quaterniond qi(qi_w, qi_xyz.x(), qi_xyz.y(), qi_xyz.z());

        // xj: [pj, qj]
        Eigen::Map<const Eigen::Vector3d> pj(xj);
        Eigen::Map<const Eigen::Vector3d> qj_xyz(xj + 3);
        const double qj_w = xj[6];
        const Eigen::Quaterniond qj(qj_w, qj_xyz.x(), qj_xyz.y(), qj_xyz.z());

        // Diferencia de posición
        Eigen::Map<Eigen::Vector3d> dp(xi_minus_xj);
        dp = pi - pj;

        // Diferencia de quaternion (rotación de qj a qi): log(qi (+) qj^{-1})
        // Aproximación de primer orden: log(q) = 2 * [qx, qy, qz] + error despreciable
        xi_minus_xj[3] = 2.0 * (-qi_w * qj_xyz.x() + qi_xyz.x() * qi_w - qi_xyz.y() * qj_xyz.z() + qi_xyz.z() * qj_xyz.y());
        xi_minus_xj[4] = 2.0 * (-qi_w * qj_xyz.y() + qi_xyz.x() * qj_xyz.z() + qi_xyz.y() * qi_w - qi_xyz.z() * qj_xyz.x());
        xi_minus_xj[5] = 2.0 * (-qi_w * qj_xyz.z() - qi_xyz.x() * qj_xyz.y() + qi_xyz.y() * qj_xyz.x() + qi_xyz.z() * qi_w);
    }

    Eigen::MatrixXd PoseLocalParameterization::boxMinusJacobianLeft(double const *xi, double const *xj) const
    {
        (void)xi;  // No usado en esta versión simplificada

        Eigen::Map<const Eigen::Vector3d> qj_xyz(xj + 3);
        const double qj_w = xj[6];
        const Eigen::Quaterniond qj(qj_w, qj_xyz.x(), qj_xyz.y(), qj_xyz.z());

        Eigen::MatrixXd J(6, 7);
        J.setZero();                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

        // Parte de posición: d(pi - pj)/dpi = I_3
        J.block<3, 3>(0, 0).setIdentity();

        // Parte de rotación: d(log(qi (+) qj^{-1}))/dqi
        // Mapeo de quaternion (7D) a vector de rotación (3D)
        // Usando matriz tipo Qleft adaptada a diferencias de quaternion
        J.block<3, 4>(3, 3) << qj_w, -qj_xyz.z(), qj_xyz.y(), -qj_xyz.x(),
                                qj_xyz.z(), qj_w, -qj_xyz.x(), -qj_xyz.y(),
                                -qj_xyz.y(), qj_xyz.x(), qj_w, -qj_xyz.z();
        J.block<3, 4>(3, 3) *= 2.0;

        return J;
    }

    Eigen::MatrixXd PoseLocalParameterization::boxMinusJacobianRight(double const *xi, double const *xj) const
    {
        (void)xj;  // No usado

        Eigen::Map<const Eigen::Vector3d> qi_xyz(xi + 3);
        const double qi_w = xi[6];
        const Eigen::Quaterniond qi(qi_w, qi_xyz.x(), qi_xyz.y(), qi_xyz.z());

        Eigen::MatrixXd J(6, 7);
        J.setZero();

        // Parte de posición: d(pi - pj)/dpj = -I_3
        J.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();

        // Parte de rotación: d(log(qi (+) qj^{-1}))/dqj
        // Derivada respecto del quaternion derecho (negativa de la izquierda por la inversión)
        J.block<3, 4>(3, 3) << -qi_w, qi_xyz.z(), -qi_xyz.y(), qi_xyz.x(),
                                -qi_xyz.z(), -qi_w, qi_xyz.x(), qi_xyz.y(),
                                qi_xyz.y(), -qi_xyz.x(), -qi_w, qi_xyz.z();
        J.block<3, 4>(3, 3) *= 2.0;

        return J;
    }

} // namespace f_vigs_slam
