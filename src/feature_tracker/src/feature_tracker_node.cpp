#include <ros/ros.h>
#include <numeric>
#include <random>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

std::default_random_engine gen;
std::normal_distribution<double> dis(0,1);

vector<double> point_num, track_len, track_rate;


// record front-end meas
// #define FE_LOG
#ifdef FE_LOG
cv::VideoWriter out_point;
int end_cnt = 1800, cnt = 0;
#endif

template<typename T>
void calstats(std::vector<T> vec, double& mean, double& var, double& min, double& size)
{
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    mean = sum / vec.size();
    var = 0.0;
    min = 1e4;
    for(int i = 0; i < vec.size(); i++)
    {
        var += pow(vec[i]-mean, 2);
        min = (min<vec[i])?min:vec[i];
    }
    var /= vec.size();
    var = sqrt(var);
    size = vec.size();   
}

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

bool init_finished = false;
void init_finished_callback(const std_msgs::BoolConstPtr &init_msg)
{
    if (init_msg->data == true)
    {
        ROS_WARN("begin to discard visual features!");
        init_finished = true;
    }
    return;
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();
    // frequency control
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    cv::Mat ret_img;
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
        ret_img = ptr->image;
    }
    else if(img_msg->encoding == "8UC3"){
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "bgr8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
        ret_img = ptr->image.clone();
        cv::cvtColor(ret_img, ret_img, cv::COLOR_BGR2GRAY);
    }
    else{
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
	    ret_img = ptr->image;
    }

    if (init_finished == true)
    {
        for (int row = 0; row < ret_img.rows; row++)
            for (int col = 0; col < ret_img.cols; col++)
            {
                if (type == 1)
                {
                    ret_img.at<uchar>(row, col) = 0;
                }
                else if (type == 2)
                {
                    ret_img.at<uchar>(row, col) = cv::saturate_cast<uchar>(0.5*pow(ret_img.at<uchar>(row, col)*0.9/255.0, 5)*255.0);
                    ret_img.at<uchar>(row, col) = cv::saturate_cast<uchar>(ret_img.at<uchar>(row, col)+dis(gen)*(1+sqrt(ret_img.at<uchar>(row, col)*0.1)));
                }
                else;
            }
    }

    cv::Mat show_img = ret_img;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ret_img.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ret_img.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ret_img.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);

        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    //draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }

            }
            //cv::imshow("vis", stereo_img);
            //cv::waitKey(5);
#ifdef FE_LOG
            cnt++;
            if (cnt <= end_cnt)
            {
                out_point << stereo_img;
                if (cnt == end_cnt)
                {
                    out_point.release();
                    exit(1);
                }
            }
#endif
            // if (init_finished == true)
            // {
            //     // cv::imwrite("/home/oran/WS/Work/SLAM/VINS-DIO/output/img/"+to_string(ulong(last_image_time*10000))+".png", stereo_img);
            //     point_num.push_back(trackerData[0].cur_pts.size());
            //     double track_cnt = 0, sum_track_len = 0;
            //     for (unsigned int j = 0; j < trackerData[0].cur_pts.size(); j++)
            //     {
            //         if (trackerData[0].track_cnt[j] > 1)
            //             track_cnt++;
            //         sum_track_len += trackerData[0].track_cnt[j];
            //         std::cout << j << ", " << trackerData[0].track_cnt[j] << ", " << sum_track_len << std::endl;
            //     }
            //     track_len.push_back(sum_track_len / trackerData[0].cur_pts.size());
            //     track_rate.push_back(track_cnt / trackerData[0].cur_pts.size());
            // }
            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

#ifdef FE_LOG
    out_point.open("/home/oran/WS/Work/SLAM/VINS-DIO/point.avi", CV_FOURCC('M', 'P', '4', '2'), 20, cv::Size(512, 512));
#endif

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);
    ros::Subscriber sub_init_finished = n.subscribe("/vins_estimator/init_finished", 2000, init_finished_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);

    ros::spin();

    std::ofstream foutA("/home/oran/WS/Work/SLAM/VINS-DIO/track.txt", std::ios::out);
    double mean = 0.0, var = 0.0, size = 0.0, min = -1.0;
    // calstats(track_rate, mean, var, size);
    // foutA << "track_rate " << mean << " " << var << " " << size << endl;
    // mean = 0.0, var = 0.0, size = 0.0;
    // calstats(track_len, mean, var, min, size);
    // foutA << "track_len " << mean << " " << var << " " << min << " " << size << endl;
    mean = 0.0, var = 0.0, size = 0.0;
    calstats(point_num, mean, var, min, size);
    foutA << "point_num " << mean << " " << var << " " << min << " " << size << endl;
    
    // std::ofstream foutA("/home/oran/WS/Work/SLAM/VINS-DIO/track-2.txt", std::ios::out);
    mean = 0.0, var = 0.0, size = 0.0;
    calstats(trackerData[0].track_rates, mean, var, min, size);
    foutA << "track_rate " << mean << " " << var << " " << min << " " << size << endl;
    mean = 0.0, var = 0.0, size = 0.0;
    calstats(trackerData[0].track_length, mean, var, min, size);
    foutA << "track_length " << mean << " " << var << " " << min << " " << size << endl;
    // for (int i = 0; i < trackerData[0].track_rates.size(); i++)
    // {
    //     foutA << trackerData[0].track_rates[i] << endl;
    // }
    foutA.close();

    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?