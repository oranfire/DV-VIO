#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "../dio_manager.h"
#include "integration_base.h"

#include <ceres/ceres.h>

class DIOVelFactor : public ceres::SizedCostFunction<3, 7, 9>
{
  public:
    DIOVelFactor() = delete;
    DIOVelFactor(IntegrationBase* _pre_integration, const NetOutput& _preds, int _intg_index, int _dio_index)
    {
        jaco_dv_dba = _pre_integration->jacobian_buf[_intg_index].block<3, 3>(O_V, O_BA);
        jaco_dv_dbg = _pre_integration->jacobian_buf[_intg_index].block<3, 3>(O_V, O_BG);
        dv = _pre_integration->dv_buf[_intg_index];
        cov_dv = _pre_integration->cov_buf[_intg_index].block<3,3>(O_V, O_V) + _preds.covs[_dio_index];
        sqrt_info = Eigen::LLT<Eigen::Matrix3d>(cov_dv.inverse()).matrixL().transpose();
        pred_v =  _preds.vels[_dio_index];
        linear_ba = _pre_integration->linearized_ba;
        linear_bg = _pre_integration->linearized_bg;
        sum_t = 0.0;
        for (int i = 0; i <= _intg_index; i++)
        {
            sum_t +=  _pre_integration->dt_buf[i];
        }
    }
    
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

        Eigen::Map<Eigen::Matrix<double, 3, 1>> residual(residuals);
        Eigen::Vector3d dba = Bai - linear_ba, dbg = Bgi - linear_bg;
        Eigen::Vector3d corrected_dv = dv + jaco_dv_dba * dba + jaco_dv_dbg * dbg;
        residual = Qi.inverse() * (G * sum_t + pred_v - Vi) - corrected_dv;
        residual = sqrt_info * residual;

        if (jacobians)
        {

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3,3>(0,3) = sqrt_info * Utility::skewSymmetric(Qi.inverse() * (G * sum_t + pred_v - Vi));

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in dio_factor jacobian[0]");
                    std::cout << sqrt_info << std::endl;
                    ROS_BREAK();
                }
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
                jacobian_speedbias_i.setZero();
                
                jacobian_speedbias_i.block<3,3>(0,0) = -Qi.inverse().toRotationMatrix();
                jacobian_speedbias_i.block<3,3>(0,3) = -jaco_dv_dba;
                jacobian_speedbias_i.block<3,3>(0,6) = -jaco_dv_dbg;
                jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

                if (jacobian_speedbias_i.maxCoeff() > 1e8 || jacobian_speedbias_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in dio_factor jacobian[1]");
                    std::cout << sqrt_info << std::endl;
                    ROS_BREAK();
                }
            }
        }

        return true;
    }

    void check(double **parameters)
    {
        double *res = new double[3];
        double **jaco = new double *[2];
        jaco[0] = new double[3 * 7];
        jaco[1] = new double[3 * 9];
        Evaluate(parameters, res, jaco);
        puts("check begins");

        puts("my");

        std::cout << Eigen::Map<Eigen::Matrix<double, 3, 1>>(res).transpose() << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>>(jaco[0]) << std::endl
                << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>>(jaco[1]) << std::endl
                << std::endl;

        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

        Eigen::Matrix<double, 3, 1> residual;
        Eigen::Vector3d dba = Bai - linear_ba, dbg = Bgi - linear_bg;
        Eigen::Vector3d corrected_dv = dv + jaco_dv_dba * dba + jaco_dv_dbg * dbg;
        residual = Qi.inverse() * (G * sum_t + pred_v - Vi) - corrected_dv;
        residual = sqrt_info * residual;

        puts("num");
        std::cout << residual.transpose() << std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 3, 12> num_jacobian;
        for (int k = 0; k < 12; k++)
        {
            Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

            Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
            Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
            Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

            if (a == 0)
                Qi = Qi * Utility::deltaQ(delta);
            else if (a == 1)
                Vi += delta;
            else if (a == 2)
                Bai += delta;
            else if (a == 3)
                Bgi += delta;

            Eigen::Matrix<double, 3, 1> tmp_residual;
            Eigen::Vector3d dba = Bai - linear_ba, dbg = Bgi - linear_bg;
            Eigen::Vector3d corrected_dv = dv + jaco_dv_dba * dba + jaco_dv_dbg * dbg;
            tmp_residual = Qi.inverse() * (G * sum_t + pred_v - Vi) - corrected_dv;
            tmp_residual = sqrt_info * tmp_residual;
            num_jacobian.col(k) = (tmp_residual - residual) / eps;
        }
        std::cout << num_jacobian << std::endl;
    }
    
    Eigen::Matrix3d jaco_dv_dba, jaco_dv_dbg, cov_dv, sqrt_info;
    double sum_t;
    Eigen::Vector3d pred_v, dv, linear_ba, linear_bg;
};

