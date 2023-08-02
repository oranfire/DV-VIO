#include "dio_manager.h"

InfoPerIMU::InfoPerIMU(double _time_stamp, const Eigen::Vector3d& _acc, const Eigen::Vector3d& _gyro, const Eigen::Matrix3d& _R)
{
    time_stamp = _time_stamp;
    acc = _acc;
    gyro = _gyro;
    R = _R;
}


#ifdef DIO_MKLOG
TrajPerIMU::TrajPerIMU(const InfoPerIMU& info, double last_time_stamp, const Eigen::Vector3d& last_P)
{
    time_stamp = info.time_stamp;
    mean_vel = info.mean_vel;
    cov_vel = info.cov_vel;
    R = info.R;
    acc = info.acc;
    gyro = info.gyro;
    if (last_time_stamp == -1)
        P = Eigen::Vector3d::Zero();
    else
        P = last_P + (time_stamp-last_time_stamp)*mean_vel;
}
#endif


DIOManager::DIOManager(std::string model_path)
{
    if (model_path == "")
    {
        std::cout << "DIO model does not exist, shut down DIO function" << std::endl;
        has_model = false;
        return;
    }
    else
        has_model = true;

    module = torch::jit::load(model_path);
    
    device_type = at::kCPU;
    if (torch::cuda::is_available())
        device_type = at::kCUDA;
    else
    {
        std::cout << "no GPU detected, program exit" << std::endl;
        exit(1);
    }
    module.to(device_type);
    // torch::set_num_threads(1);

    for (int i = 0; i < AUG_N; i++)
    {
        double theta = 2*M_PI/AUG_N*i;
        Eigen::AngleAxisd yawAngle(theta, Eigen::Vector3d::UnitZ());
        aug_Rs.push_back(yawAngle.toRotationMatrix());
        // aug_Rs.push_back(Eigen::Matrix3d::Identity());
    }

    dio_process = std::thread(&DIOManager::process, this);
}

void DIOManager::inputData(double time_stamp, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc, const Eigen::Matrix3d& R)
{
    if (has_model == false)
        return;

    mutex_datas.lock();
    if (datas.size() == 0 || abs(time_stamp-datas.back().time_stamp-0.01) < 0.001)
    {
        Eigen::Vector3d r_acc = R*acc-Eigen::Vector3d(0,0,9.805), r_gyro = R*gyro;
        datas.push_back(InfoPerIMU(time_stamp, r_acc, r_gyro, R));
    }
    mutex_datas.unlock();
}

void DIOManager::removeData()
{
    if (has_model == false || datas.size() <= 2*NET_IMU_CNT)
        return;

    mutex_datas.lock();
    int dist = datas.size() - 2*NET_IMU_CNT;
#ifdef DIO_MKLOG
    for (int i = 0; i < dist; i++)
    {
        if (trajs.size() == 0)
            if (datas[i].vels.size() == 0)
                continue;
            else
                trajs.push_back(TrajPerIMU(datas[i], -1, Eigen::Vector3d::Zero()));
        else 
            if (datas[i].vels.size() == 0)
                std::cout << "problem!: no pred. vel found!" << std::endl;
            else
                trajs.push_back(TrajPerIMU(datas[i], trajs.back().time_stamp, trajs.back().P));
    }
#endif
    for (int i = 0; i < 2*NET_IMU_CNT; i++)
    {
        datas[i] = datas[i+dist];
    }
    datas.resize(2*NET_IMU_CNT);
    mutex_datas.unlock();
    
    // printStatus();
}

void DIOManager::triggerInfer()
{
    if (has_model == false || datas.size() < NET_IMU_CNT)
        return;
    
    mutex_datas.lock();
    NetInput in;
    for (int i = datas.size()-NET_IMU_CNT; i < datas.size(); i++)
    {
        in.time_stamp.push_back(datas[i].time_stamp);
        in.accs.push_back(datas[i].acc);
        in.gyros.push_back(datas[i].gyro);
    }
    mutex_datas.unlock();
    
    mutex_in_buf.lock();
    input_buf.push(in);
    mutex_in_buf.unlock();
    con_in_buf.notify_one();
}

std::vector<NetInput> DIOManager::getNetInputs()
{
    std::vector<NetInput> ins;
    while (input_buf.size() != 0)
    {
        ins.push_back(input_buf.back());
        input_buf.pop();
    }
    return ins;
}

void DIOManager::process()
{
    while (true)
    {
        std::vector<NetInput> ins;
        std::unique_lock<std::mutex> lk(mutex_in_buf);
        con_in_buf.wait(lk, [&]
                 {
            return (ins = getNetInputs()).size() != 0;
                 });
        lk.unlock();
        
        for (int i = 0; i < ins.size(); i++)
        {
            TicToc t;
            Eigen::Matrix<double,3,NET_IMU_CNT> gyro, acc;
            for (int j = 0; j < NET_IMU_CNT; j++)
            {
                acc.col(j) = ins[i].accs[j];
                gyro.col(j) = ins[i].gyros[j];
            }
            
            std::vector<Eigen::Matrix<float,3,NET_IMU_CNT>> aug_accs, aug_gyros; // ColMajor
            Eigen::AngleAxisd rand_rot(rand()%100/50.0*M_PI, Eigen::Vector3d::UnitZ());
            // Eigen::AngleAxisd rand_rot(0, Eigen::Vector3d::UnitZ());
            for (int j = 0; j < AUG_N; j++)
            {    
                aug_accs.push_back((rand_rot.matrix()*aug_Rs[j]*acc).cast<float>());
                aug_gyros.push_back((rand_rot.matrix()*aug_Rs[j]*gyro).cast<float>());
            }

            std::vector<at::Tensor> imus_tensor;
            for (int j = 0; j < AUG_N; j++)
            {
                at::Tensor acc_tensor = torch::from_blob(aug_accs[j].data(), {NET_IMU_CNT,3});
                at::Tensor gyro_tensor = torch::from_blob(aug_gyros[j].data(), {NET_IMU_CNT,3});
                at::Tensor imu_tensor = torch::cat({gyro_tensor, acc_tensor}, 1);
                imus_tensor.push_back(imu_tensor);    
            }
#if AUG_N == 8
            at::Tensor input_tensor = torch::stack({imus_tensor[0], imus_tensor[1], imus_tensor[2], imus_tensor[3], imus_tensor[4], imus_tensor[5], imus_tensor[6], imus_tensor[7]}, 0);
#else
#error this is a compile time error message.
#endif
            std::cout << "input preprocess time: " << t.toc() << " ms" << std::endl; 

            t.tic();
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor.to(device_type));
            std::cout << "input transfer time: " << t.toc() << " ms" << std::endl;

            t.tic();
            at::Tensor output_tensor = module.forward(inputs).toTensor().to(at::kCPU).reshape(-1); // reshape is required for correct order of storage in memory
            std::cout << "model forward time: " << t.toc() << " ms" << std::endl;

            t.tic();
            mutex_datas.lock();
            int index_imu = 0;
            for (int j = 0; j < NET_IMU_CNT; j++)
            {
                if (datas[index_imu].time_stamp > ins[i].time_stamp[j])
                    continue;
                while (datas[index_imu].time_stamp < ins[i].time_stamp[j])
                    index_imu++;
                for (int k = 0; k < AUG_N; k++)
                {
                    std::vector<float> aug_vel_v(output_tensor.data_ptr<float>()+(k*NET_IMU_CNT+j)*6, output_tensor.data_ptr<float>()+(k*NET_IMU_CNT+j)*6+3);
                    Eigen::Vector3f aug_vel = Eigen::Map<Eigen::Vector3f>(aug_vel_v.data());
                    Eigen::Vector3d vel = aug_Rs[k].transpose()*rand_rot.matrix().transpose()*aug_vel.cast<double>();
                    // std::cout << "aug_vel: " << aug_vel.transpose() << std::endl;
                    datas[index_imu].vels.push_back(vel);
                }
            }
            mutex_datas.unlock();
            std::cout << "model post-process time: " << t.toc() << " ms" << std::endl;
        }
    }
}

NetOutput DIOManager::buildVelFactor(double time_start, double time_end)
{
    NetOutput out;
    mutex_datas.lock();
    for (int i = 0; i < datas.size(); i++)
    {
        if (datas[i].vels.size() == 0 || datas[i].time_stamp < time_start || datas[i].time_stamp > time_end)
            continue;
        datas[i].mean_vel = Eigen::Vector3d::Zero();
        for (int j = 0; j < datas[i].vels.size(); j++)
        {
            datas[i].mean_vel += datas[i].vels[j]; 
        }
        datas[i].mean_vel /= datas[i].vels.size();
        datas[i].cov_vel = Eigen::Matrix3d::Zero();
        for (int j = 0; j < datas[i].vels.size(); j++)
        {
            Eigen::Vector3d delta_vel = datas[i].vels[j]-datas[i].mean_vel;
            datas[i].cov_vel += delta_vel*delta_vel.transpose(); 
        }
        datas[i].cov_vel /= datas[i].vels.size();
        out.time_stamp.push_back(datas[i].time_stamp);
        out.vels.push_back(datas[i].mean_vel);
        out.covs.push_back(datas[i].cov_vel);
    }
    mutex_datas.unlock();
    return out;
} 

void DIOManager::printStatus()
{
    mutex_datas.lock();
    for (int i = 0; i < datas.size(); i++)
    {
        std::cout << "data[" << i << "], time_stamp: " << datas[i].time_stamp << ", preds_num: " << datas[i].vels.size() << std::endl;
        if (datas[i].vels.size() == 0)
            continue;
        std::cout << "mean_vel: " << datas[i].mean_vel.transpose() << std::endl;
        std::cout << "cov_vel: " << datas[i].cov_vel << std::endl;
    }
    mutex_datas.unlock();
} 
