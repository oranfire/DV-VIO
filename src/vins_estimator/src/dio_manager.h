#pragma once

#include "parameters.h"
#include "utility/tic_toc.h"
#include <unordered_map>
#include <queue>
#include <torch/torch.h>
#include <torch/script.h>

#define NET_IMU_CNT 560
#define USE_TTA 
#define AUG_N 8

class NetOutput
{
public:
    std::vector<double> time_stamp;
    std::vector<Eigen::Vector3d> vels;
    std::vector<Eigen::Matrix3d> covs;
};

class NetInput
{
public:
    std::vector<double> time_stamp;
    std::vector<Eigen::Vector3d> accs, gyros;
};

class InfoPerIMU
{
public:
    InfoPerIMU(double _time_stamp, const Eigen::Vector3d& _acc, const Eigen::Vector3d& _gyro, const Eigen::Matrix3d& _R);
    InfoPerIMU() {}

    double time_stamp;
    Eigen::Vector3d acc, gyro;
    Eigen::Matrix3d R;

    std::vector<Eigen::Vector3d> vels;
    std::vector<Eigen::Matrix3d> covs;
    Eigen::Vector3d mean_vel;
    Eigen::Matrix3d cov_vel;
};

#ifdef DIO_MKLOG
class TrajPerIMU
{
public:
    TrajPerIMU(const InfoPerIMU& info, double last_time_stamp, const Eigen::Vector3d& last_P);

    double time_stamp;
    Eigen::Vector3d acc, gyro;
    Eigen::Matrix3d R;
    Eigen::Vector3d mean_vel, P;
    Eigen::Matrix3d cov_vel;
};
#endif

class DIOManager
{
  public:
    DIOManager() {}
    DIOManager(std::string model_path);

    void inputData(double time_stamp, const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc, const Eigen::Matrix3d& R);
    void removeData();

    void triggerInfer();

    std::vector<NetInput> getNetInputs();
    void process();

    NetOutput buildVelFactor(double time_start=-DBL_MAX, double time_end=DBL_MAX); 

    void printStatus();   

    torch::jit::script::Module module;
    bool has_model;
    torch::DeviceType device_type;

    std::thread dio_process;

    std::vector<InfoPerIMU> datas;
#ifdef DIO_MKLOG
    std::vector<TrajPerIMU> trajs;
#endif
    std::mutex mutex_datas;

    std::queue<NetInput> input_buf;
    std::mutex mutex_in_buf;
    std::condition_variable con_in_buf;

    std::vector<Eigen::Matrix3d> aug_Rs;
};
