#include "f_vigs_slam/ImuCostFunction.hpp"

namespace f_vigs_slam
{
    namespace
    {
        /**
        * @brief Devuelve la matriz antisimetrica asociada a un vector 3D
        *
        *   Corresponde a [v]_x segun la notacion que seguimos en el trabajo
        *   Esta matriz cumple que [v]_x * w = v x w (producto vectorial)
        **/
        inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v)
        {
            Eigen::Matrix3d m;
            m << 0.0, -v.z(), v.y(),
                 v.z(), 0.0, -v.x(),
                -v.y(), v.x(), 0.0;
            return m;
        }

        /**
        * @brief Aproximacion de primer orden de la exponencial de SO(3) para rotaciones pequeñas
        *
        **/
        inline Eigen::Quaterniond deltaQ(const Eigen::Vector3d &theta)
        {
            Eigen::Quaterniond dq;
            Eigen::Vector3d half = 0.5 * theta;
            dq.w() = 1.0;
            dq.x() = half.x();
            dq.y() = half.y();
            dq.z() = half.z();
            return dq;
        }

        /** 
        * @brief Devuelve una matriz para representar suma a izquierda de rotaciones
        *
        *  q (+) p = Qleft(q) * p con "(+)" la suma a izquierda en SO(3)
        *
        **/
        inline Eigen::Matrix4d Qleft(const Eigen::Quaterniond &q)
        {

            Eigen::Matrix4d m;
            const double w = q.w();
            const double x = q.x();
            const double y = q.y();
            const double z = q.z();
            m << w, -x, -y, -z,
                 x,  w, -z,  y,
                 y,  z,  w, -x,
                 z, -y,  x,  w;
            return m;
        }

        /** 
         * @brief Devuelve una matriz para representar suma a derecha de rotaciones
         *
         *  p (+) q = Qright(q) * p con "(+)" la suma a derecha en SO(3)
         *
        **/
        inline Eigen::Matrix4d Qright(const Eigen::Quaterniond &q)
        {
            Eigen::Matrix4d m;
            const double w = q.w();
            const double x = q.x();
            const double y = q.y();
            const double z = q.z();
            m << w, -x, -y, -z,
                 x,  w,  z, -y,
                 y, -z,  w,  x,
                 z,  y, -x,  w;
            return m;
        }
    }

    ImuCostFunction::ImuCostFunction(std::shared_ptr<Preintegration> preintegration)
        : pre_integration_(std::move(preintegration))
    {
    }

    /**
    * @brief Evalua el valor de la funcion de coste y sus jacobianos
    *
    * 
    **/
    bool ImuCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Vector3d> Pi(parameters[0]);
        Eigen::Map<const Eigen::Quaterniond> Qi(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> Vi(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> Bai(parameters[1] + 3);
        Eigen::Map<const Eigen::Vector3d> Bgi(parameters[1] + 6);

        Eigen::Map<const Eigen::Vector3d> Pj(parameters[2]);
        Eigen::Map<const Eigen::Quaterniond> Qj(parameters[2] + 3);
        Eigen::Map<const Eigen::Vector3d> Vj(parameters[3]);
        Eigen::Map<const Eigen::Vector3d> Baj(parameters[3] + 3);
        Eigen::Map<const Eigen::Vector3d> Bgj(parameters[3] + 6);

        // Si hubo cambios significativos en los sesgos, volvemos a integrar pero con los sesgos nuevos
        if ((Bai - pre_integration_->ba).norm() > 0.10 ||
            (Bgi - pre_integration_->bg).norm() > 0.01)
        {
            pre_integration_->repropagate(Bai, Bgi);
        }

        // Calculamos el residual
        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        residual = pre_integration_->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                               Pj, Qj, Vj, Baj, Bgj);

    
        // REPASAR LA MATRIZ DE INFORMACION

        // Obtenemos la informacion de la medicion para pesar el residual
        // El objetivo es minimizar la siguiente funcion de coste:
        // ||r||^2 = r^T * W * r con W = cov^-1
        // Si multiplicamos el residual por sqrt(W) obtenemos:
        // ||r'||^2 = r'^T * r' con r' = sqrt(W) * r
        Eigen::Matrix<double, 15, 15> sqrt_info =
            Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration_->covariance.inverse())
                .matrixL()
                .transpose();

        residual = sqrt_info * residual;

        // Si el solver pide los jacobianos, los calculamos
        if (jacobians)
        {
            // Obtenemos los jacobianos de la preintegracion de la posicion p, la rotacion q y la velocidad v 
            // respecto del sesgo del acelerometro ba y del giroscopio bg
            // Esto corresponde arepresentar el estado como: x = [p, q, v, ba, bg]
            // Obs: no tenemos dq_dba ya que la rotacion no depende de la aceleracion

            const double sum_dt = pre_integration_->sum_dt;

            const Eigen::Matrix3d dp_dba = pre_integration_->jacobian.block<3, 3>(0, 9);
            const Eigen::Matrix3d dp_dbg = pre_integration_->jacobian.block<3, 3>(0, 12);

            const Eigen::Matrix3d dq_dbg = pre_integration_->jacobian.block<3, 3>(3, 12);

            const Eigen::Matrix3d dv_dba = pre_integration_->jacobian.block<3, 3>(6, 9);
            const Eigen::Matrix3d dv_dbg = pre_integration_->jacobian.block<3, 3>(6, 12);


            // En este termino tenemos el jacobiano del residual respecto de la pose en el frame i
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J_pose_i(jacobians[0]);
                J_pose_i.setZero();

                J_pose_i.block<3, 3>(0, 0) = -Qi.inverse().toRotationMatrix();
                J_pose_i.block<3, 3>(0, 3) = skewSymmetric(Qi.inverse() * (0.5 * G_ * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

                Eigen::Quaterniond corrected_delta_q = pre_integration_->delta_q * deltaQ(dq_dbg * (Bgi - pre_integration_->bg));
                J_pose_i.block<3, 3>(3, 3) = -(Qleft(Qj.inverse() * Qi) * Qright(corrected_delta_q)).bottomRightCorner<3, 3>();

                J_pose_i.block<3, 3>(6, 3) = skewSymmetric(Qi.inverse() * (G_ * sum_dt + Vj - Vi));

                J_pose_i = sqrt_info * J_pose_i;
            }

            // En este termino tenemos el jacobiano del residual respecto de la velocidad y sesgos
            // en el frame i
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J_speedbias_i(jacobians[1]);
                J_speedbias_i.setZero();

                J_speedbias_i.block<3, 3>(0, 0) = -Qi.inverse().toRotationMatrix() * sum_dt;
                J_speedbias_i.block<3, 3>(0, 3) = -dp_dba;
                J_speedbias_i.block<3, 3>(0, 6) = -dp_dbg;

                J_speedbias_i.block<3, 3>(3, 6) = -Qleft(Qj.inverse() * Qi * pre_integration_->delta_q)
                                                     .bottomRightCorner<3, 3>() * dq_dbg;

                J_speedbias_i.block<3, 3>(6, 0) = -Qi.inverse().toRotationMatrix();
                J_speedbias_i.block<3, 3>(6, 3) = -dv_dba;
                J_speedbias_i.block<3, 3>(6, 6) = -dv_dbg;

                J_speedbias_i.block<3, 3>(9, 3) = -Eigen::Matrix3d::Identity();
                J_speedbias_i.block<3, 3>(12, 6) = -Eigen::Matrix3d::Identity();

                J_speedbias_i = sqrt_info * J_speedbias_i;
            }

            // En este termino tenemos el jacobiano del residual respecto de la pose en el frame j
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J_pose_j(jacobians[2]);
                J_pose_j.setZero();

                J_pose_j.block<3, 3>(0, 0) = Qi.inverse().toRotationMatrix();

                Eigen::Quaterniond corrected_delta_q = pre_integration_->delta_q * deltaQ(dq_dbg * (Bgi - pre_integration_->bg));
                J_pose_j.block<3, 3>(3, 3) = Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();

                J_pose_j = sqrt_info * J_pose_j;
            }

            // En este termino tenemos el jacobiano del residual respecto de la velocidad y sesgos
            // en el frame j
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J_speedbias_j(jacobians[3]);
                J_speedbias_j.setZero();

                J_speedbias_j.block<3, 3>(6, 0) = Qi.inverse().toRotationMatrix();
                J_speedbias_j.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity();
                J_speedbias_j.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();

                J_speedbias_j = sqrt_info * J_speedbias_j;
            }
        }

        return true;
    }
}