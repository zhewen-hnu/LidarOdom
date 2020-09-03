#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <mutex>
#include <queue>
#include <fstream>
#include <iostream>

#include "../include/utils/common.h"
#include "../include/utils/TicToc.h"
#include "../include/factor/lidarFactor.hpp"
#include "../include/utils/Twist.h"
#include "../include/utils/geometry_utils.h"
#include "../include/utils/CircularBuffer.h"
#include "../include/utils/math_utils.h"


using namespace geometryutils;
using namespace lom;

typedef Twist<double> Transform;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;

#define DISTORTION 0

int skip_fream = 0;
int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 1;
constexpr double NEARBY_SCAN = 2.5;

int skipFrameNum = 5;
bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;
double timeSurfPointsFlat_z_rot_xy_trans = 0;
double timeSurfPointsFlat_z_trans = 0;
double timeSurfPointsFlat_x_rot = 0;
double timeSurfPointsFlat_y_rot = 0;
double timeSurfPointsFlatNorm = 0;

double dis = 0.0;

bool together_opt = true;
bool using_sharp_point = true;
bool using_flat_point = true;
bool using_last_sharp_point = true;
bool using_last_flat_point = true;
bool using_local_map = true;
bool using_full_point = true;
bool first_fream = false;
bool key_fream = false;

Eigen::Affine3d transf_se_ = Eigen::Affine3d::Identity();
Eigen::Affine3d transf_sum_ = Eigen::Affine3d::Identity();
Eigen::Affine3d transf_sum_last = Eigen::Affine3d::Identity();

CircularBuffer<PointCloud> local_map_flat(20);
CircularBuffer<PointCloud> local_map_sharp(20);
CircularBuffer<PointCloud> local_map_less_flat(20);
CircularBuffer<PointCloud> local_map_less_sharp(20);
CircularBuffer<Eigen::Affine3d> trans_sum_for_local_map(20);

pcl::KdTreeFLANN<PointT>::Ptr kdtree_corner_(new pcl::KdTreeFLANN<PointT>());
pcl::KdTreeFLANN<PointT>::Ptr kdtree_surf_(new pcl::KdTreeFLANN<PointT>());
pcl::KdTreeFLANN<PointT>::Ptr kdtree_full_(new pcl::KdTreeFLANN<PointT>());

pcl::KdTreeFLANN<PointT>::Ptr kdtree_corner_last_(new pcl::KdTreeFLANN<PointT>());
pcl::KdTreeFLANN<PointT>::Ptr kdtree_surf_last_(new pcl::KdTreeFLANN<PointT>());
pcl::KdTreeFLANN<PointT>::Ptr kdtree_full_last_(new pcl::KdTreeFLANN<PointT>());

pcl::PointCloud<PointT>::Ptr full_ponits_ptr(new pcl::PointCloud<PointT>());
pcl::PointCloud<PointT>::Ptr corner_points_sharp_ptr(new pcl::PointCloud<PointT>());
pcl::PointCloud<PointT>::Ptr corner_points_less_sharp_ptr(new pcl::PointCloud<PointT>());
pcl::PointCloud<PointT>::Ptr surface_points_flat_ptr(new pcl::PointCloud<PointT>());
pcl::PointCloud<PointT>::Ptr surface_points_less_flat_ptr(new pcl::PointCloud<PointT>());
pcl::PointCloud<PointT>::Ptr surfPointsLessFlatMap(new pcl::PointCloud<PointT>());

pcl::PointCloud<PointT>::Ptr corner_points_sharp_last_ptr(new pcl::PointCloud<PointT>());
pcl::PointCloud<PointT>::Ptr surface_points_flat_last_ptr(new pcl::PointCloud<PointT>());
pcl::PointCloud<PointT>::Ptr corner_points_less_sharp_last_ptr(new pcl::PointCloud<PointT>());
pcl::PointCloud<PointT>::Ptr surface_points_less_flat_last_ptr(new pcl::PointCloud<PointT>());
pcl::PointCloud<PointT>::Ptr full_ponits_last_ptr(new pcl::PointCloud<PointT>());


pcl::PointCloud<PointT>::Ptr surfPointsFlatNorm(new pcl::PointCloud<PointT>());

// Transformation from current frame to world frame
// Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
// Eigen::Vector3d t_w_curr(0, 0, 0);


// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

double transf_sum_para_q[4] = {0, 0, 0, 1};
double transf_sum_para_t[3] = {0, 0, 0};

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

Eigen::Map<Eigen::Quaterniond> q_w_curr(transf_sum_para_q);
Eigen::Map<Eigen::Vector3d> t_w_curr(transf_sum_para_t);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;

std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatNormBuf;

std::mutex mBuf;

// undistort lidar point
void TransformToStart(PointT const *const pi, PointT *const po)
{
    //interpolation ratio
    double s;
    if (DISTORTION)
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    //s = 1;
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    Eigen::Vector3d t_point_last = s * t_last_curr;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

// transform all lidar points to the start of the next frame

void TransformToEnd(PointT const *const pi, PointT *const po)
{
    // undistort point first
    PointT un_point_tmp;
    TransformToStart(pi, &un_point_tmp);

    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = int(pi->intensity);
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

void laserCloudFlatNormHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlatNorm2)
{
    mBuf.lock();
    surfFlatNormBuf.push(surfPointsFlatNorm2);
    mBuf.unlock();
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;

    nh.param("mapping_skip_frame", skipFrameNum, 2);
    nh.param("together_opt", together_opt, true);
    nh.param("using_sharp_point", using_sharp_point, true);
    nh.param("using_flat_point", using_flat_point, true);
    nh.param("using_last_sharp_point", using_last_sharp_point, true);
    nh.param("using_last_flat_point", using_last_flat_point, true);
    nh.param("using_local_map", using_local_map, true);
    nh.param("using_full_point", using_full_point, true);

    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);

    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);

    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);

    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/full_cloud", 100, laserCloudFullResHandler);

    ros::Subscriber subSurfPointsFlatNorm = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat_normal", 100, laserCloudFlatNormHandler);

    ros::Publisher pubKeyFreamPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/key_fream_cloud_less_flat", 1000000);

    ros::Publisher pubKeyFreamPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/key_fream_cloud_less_sharp", 1000000);

    ros::Publisher pubKeyFreamPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/key_fream_cloud_flat", 1000000);

    ros::Publisher pubKeyFreamPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/key_fream_cloud_sharp", 1000000);

    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    ros::Publisher pubLocalMappingLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/local_mapping_sharp", 100);

    ros::Publisher pubLocalMappingLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/local_mapping_flat", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

    ros::Publisher pubKeyFreamOdometry = nh.advertise<nav_msgs::Odometry>("/key_fream_to_init", 1000000);

    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);

    ros::Publisher pubLaserOdometry_last_curr = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_last", 1);

    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    nav_msgs::Path laserPath;

    int frameCount = 0;
    ros::Rate rate(100);

    while (ros::ok())
    {
        ros::spinOnce();

        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
            !fullPointsBuf.empty())
        {

            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();
            
            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                printf("unsync messeage!");
                ROS_BREAK();
            }

            mBuf.lock();
            corner_points_sharp_ptr->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *corner_points_sharp_ptr);
            cornerSharpBuf.pop();

            corner_points_less_sharp_ptr->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *corner_points_less_sharp_ptr);
            cornerLessSharpBuf.pop();

            surface_points_flat_ptr->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surface_points_flat_ptr);
            surfFlatBuf.pop();

            surface_points_less_flat_ptr->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surface_points_less_flat_ptr);
            surfLessFlatBuf.pop();

            full_ponits_ptr->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *full_ponits_ptr);
            fullPointsBuf.pop();

            mBuf.unlock();            

            TicToc t_whole;
            // initializing
            if (!systemInited)
            {
                first_fream = true;
                systemInited = true; 
                std::cout << "Initialization finished \n";
            }
            else
            {

                int corner_points_sharp_num = corner_points_sharp_ptr->points.size();
                int surface_points_flat_num = surface_points_flat_ptr->points.size();
                int corner_points_sharp_last_num = corner_points_sharp_last_ptr->size();
                int surface_points_flat_last_num = surface_points_flat_last_ptr->size();
		int full_ponits_num = full_ponits_ptr->points.size();


                TicToc t_opt;

                kdtree_corner_->setInputCloud(corner_points_less_sharp_ptr);
                kdtree_surf_->setInputCloud(surface_points_less_flat_ptr);
		kdtree_full_->setInputCloud(full_ponits_ptr);
		
// 		kdtree_corner_->setInputCloud(corner_points_less_sharp_ptr);
// 		kdtree_surf_->setInputCloud(surface_points_less_flat_ptr);
// 		kdtree_full_last_->setInputCloud(full_ponits_ptr);


                for (size_t opti_counter = 0; opti_counter < 5; ++opti_counter)
                {
                    corner_correspondence = 0;
                    plane_correspondence = 0;

                    //ceres::LossFunction *loss_function = NULL;
//                     ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    ceres::LossFunction *loss_function = new ceres::CauchyLoss(0.1);
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;

                    ceres::Problem problem(problem_options);
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                    problem.AddParameterBlock(para_t, 3);

                    PointT point_sel;
                    std::vector<int> point_search_ind;
                    std::vector<float> point_search_sq_dis;

                    TicToc t_data;

                    if (using_sharp_point == true)
                    {
                        PointCloud transformed_cloud;
                        pcl::transformPointCloud(*corner_points_sharp_ptr, transformed_cloud, transf_se_);
                        for (int i = 0; i < corner_points_sharp_num; ++i)
                        {
                            point_sel = transformed_cloud[i];
                            int change_ind = 1;
                            PointT point_ori = corner_points_sharp_ptr->points[i];
                            double sqrt_dis = point_ori.x * point_ori.x + point_ori.y * point_ori.y + point_ori.z * point_ori.z;

                            kdtree_corner_last_->nearestKSearch(point_sel, 6, point_search_ind, point_search_sq_dis);

                            int first_point_ring = int(corner_points_less_sharp_last_ptr->points[point_search_ind[0]].intensity);
                            int second_point_ring = int(corner_points_less_sharp_last_ptr->points[point_search_ind[1]].intensity);

                            if (first_point_ring == second_point_ring) {
                                for (int n = 2; n < 6; ++n) {
                                    int next_point_ring = int(corner_points_less_sharp_last_ptr->points[point_search_ind[n]].intensity);
                                    if (second_point_ring != next_point_ring) {
                                        second_point_ring = next_point_ring;
                                        int ind_temp = point_search_ind[n];
                                        float sq_dis_temp = point_search_sq_dis[n];
                                        point_search_ind[n] = point_search_ind[1];
                                        point_search_sq_dis[n] = point_search_sq_dis[1];
                                        point_search_ind[1] = ind_temp;
                                        point_search_sq_dis[1] = sq_dis_temp;
                                        change_ind = n;
                                        break;

                                    }
                                }
                            }

                            if (first_point_ring != second_point_ring && point_search_sq_dis[1] < DISTANCE_SQ_THRESHOLD) {
                                Eigen::Vector3d curr_point(corner_points_sharp_ptr->points[i].x,
                                                        corner_points_sharp_ptr->points[i].y,
                                                        corner_points_sharp_ptr->points[i].z);
                                Eigen::Vector3d last_point_a(corner_points_less_sharp_last_ptr->points[point_search_ind[0]].x,
                                                            corner_points_less_sharp_last_ptr->points[point_search_ind[0]].y,
                                                            corner_points_less_sharp_last_ptr->points[point_search_ind[0]].z);
                                Eigen::Vector3d last_point_b(corner_points_less_sharp_last_ptr->points[point_search_ind[1]].x,
                                                            corner_points_less_sharp_last_ptr->points[point_search_ind[1]].y,
                                                            corner_points_less_sharp_last_ptr->points[point_search_ind[1]].z);

                                double s;
                                if (DISTORTION)
                                    s = (point_ori.intensity - int(point_ori.intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;

                                ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                                //ceres::CostFunction *cost_function = new LidarEdgeFactor_z_rot_xy_trans(curr_point, last_point_a, last_point_b, s);
                                //ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, last_point_a);
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                corner_correspondence++;
                            } else if (change_ind == 1) {
                                if (point_search_sq_dis[4] < 0.01 * sqrt_dis) {
                                    Eigen::Vector3d center(0, 0, 0);
                                    for (int j = 0; j < 5; j++)
                                    {
                                        Eigen::Vector3d tmp(corner_points_less_sharp_last_ptr->points[point_search_ind[j]].x,
                                                            corner_points_less_sharp_last_ptr->points[point_search_ind[j]].y,
                                                            corner_points_less_sharp_last_ptr->points[point_search_ind[j]].z);
                                        center = center + tmp;
                                    }
                                    center = center / 5.0;
                                    Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                                    ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
                                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                }
                            } else {
                                int ind_temp = point_search_ind[change_ind];
                                float sq_dis_temp = point_search_sq_dis[change_ind];
                                point_search_ind[change_ind] = point_search_ind[1];
                                point_search_sq_dis[change_ind] = point_search_sq_dis[1];
                                point_search_ind[1] = ind_temp;
                                point_search_sq_dis[1] = sq_dis_temp;
                                if (point_search_sq_dis[4] < 0.01 * sqrt_dis) {
                                    Eigen::Vector3d center(0, 0, 0);
                                    for (int j = 0; j < 5; j++)
                                    {
                                        Eigen::Vector3d tmp(corner_points_less_sharp_last_ptr->points[point_search_ind[j]].x,
                                                            corner_points_less_sharp_last_ptr->points[point_search_ind[j]].y,
                                                            corner_points_less_sharp_last_ptr->points[point_search_ind[j]].z);
                                        center = center + tmp;
                                    }
                                    center = center / 5.0;
                                    Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                                    ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
                                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                }
                            }

                        }
                    }



                // find correspondence for plane features

                    if (using_flat_point == true)
                    {
                        PointCloud transformed_cloud;
                        pcl::transformPointCloud(*surface_points_flat_ptr, transformed_cloud, transf_se_);

                        for (int i = 0; i < surface_points_flat_num; ++i)
                        {
                            point_sel = transformed_cloud[i];
                            kdtree_surf_last_->nearestKSearch(point_sel, 6, point_search_ind, point_search_sq_dis);
                            if (point_search_sq_dis[0] < DISTANCE_SQ_THRESHOLD) {

                                int first_point_ring = int(surface_points_less_flat_last_ptr->points[point_search_ind[0]].intensity);
                                int second_point_ring = int(surface_points_less_flat_last_ptr->points[point_search_ind[1]].intensity);
                                int third_point_ring = int(surface_points_less_flat_last_ptr->points[point_search_ind[2]].intensity);

                                if (first_point_ring == second_point_ring && second_point_ring == third_point_ring) {
                                    for (int n = 3; n < 6; ++n) {
                                        int next_point_ring = int(surface_points_less_flat_last_ptr->points[point_search_ind[n]].intensity);
                                        if (third_point_ring != next_point_ring) {
                                            third_point_ring = next_point_ring;
                                            point_search_ind[2] = point_search_ind[n];
                                            point_search_sq_dis[2] = point_search_sq_dis[n];
                                            break;
                                        }
                                    }

                                    if (second_point_ring == third_point_ring) {
                                        continue;
                                    }
                                }
                                if (point_search_sq_dis[2] < DISTANCE_SQ_THRESHOLD) {
                                    Eigen::Vector3d curr_point(surface_points_flat_ptr->points[i].x,
                                                               surface_points_flat_ptr->points[i].y,
                                                               surface_points_flat_ptr->points[i].z);
                                    Eigen::Vector3d last_point_a(surface_points_less_flat_last_ptr->points[point_search_ind[0]].x,
                                                                 surface_points_less_flat_last_ptr->points[point_search_ind[0]].y,
                                                                 surface_points_less_flat_last_ptr->points[point_search_ind[0]].z);
                                    Eigen::Vector3d last_point_b(surface_points_less_flat_last_ptr->points[point_search_ind[1]].x,
                                                                 surface_points_less_flat_last_ptr->points[point_search_ind[1]].y,
                                                                 surface_points_less_flat_last_ptr->points[point_search_ind[1]].z);
                                    Eigen::Vector3d last_point_c(surface_points_less_flat_last_ptr->points[point_search_ind[2]].x,
                                                                 surface_points_less_flat_last_ptr->points[point_search_ind[2]].y,
                                                                 surface_points_less_flat_last_ptr->points[point_search_ind[2]].z);

                                    double s;
                                    if (DISTORTION)
                                        s = (surface_points_flat_ptr->points[i].intensity - int(surface_points_flat_ptr->points[i].intensity)) / SCAN_PERIOD;
                                    else
                                        s = 1.0;
                                    ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
// 				    ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                    //ceres::CostFunction *cost_function = new LidarPlaneFactor_z_rot_xy_trans(curr_point, last_point_a, last_point_b, last_point_c, s);
                                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                    plane_correspondence++;                                    
                                }
                            }

                        }
                    }



                    if (using_last_sharp_point == true)
                    {
                        PointCloud transformed_cloud;
                        pcl::transformPointCloud(*corner_points_sharp_last_ptr, transformed_cloud, transf_se_.inverse());

                        for (int i = 0; i < corner_points_sharp_last_num; ++i)
                        {

                            point_sel = transformed_cloud[i];
                            int change_ind = 1;
                            PointT point_ori = corner_points_sharp_last_ptr->points[i];
                            double sqrt_dis = point_ori.x * point_ori.x + point_ori.y * point_ori.y + point_ori.z * point_ori.z;

                            kdtree_corner_->nearestKSearch(point_sel, 6, point_search_ind, point_search_sq_dis);

                            int first_point_ring = int(corner_points_less_sharp_ptr->points[point_search_ind[0]].intensity);
                            int second_point_ring = int(corner_points_less_sharp_ptr->points[point_search_ind[1]].intensity);

                            if (first_point_ring == second_point_ring) {
                                for (int n = 2; n < 6; ++n) {
                                    int next_point_ring = int(corner_points_less_sharp_ptr->points[point_search_ind[n]].intensity);
                                    if (second_point_ring != next_point_ring) {
                                        second_point_ring = next_point_ring;
                                        int ind_temp = point_search_ind[n];
                                        float sq_dis_temp = point_search_sq_dis[n];
                                        point_search_ind[n] = point_search_ind[1];
                                        point_search_sq_dis[n] = point_search_sq_dis[1];
                                        point_search_ind[1] = ind_temp;
                                        point_search_sq_dis[1] = sq_dis_temp;
                                        change_ind = n;
                                        break;

                                    }
                                }
                            }

                            if (first_point_ring != second_point_ring && point_search_sq_dis[1] < DISTANCE_SQ_THRESHOLD) {
                                Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                                Eigen::Vector3d last_point_a(corner_points_less_sharp_ptr->points[point_search_ind[0]].x,
                                                             corner_points_less_sharp_ptr->points[point_search_ind[0]].y,
                                                             corner_points_less_sharp_ptr->points[point_search_ind[0]].z);
                                Eigen::Vector3d last_point_b(corner_points_less_sharp_ptr->points[point_search_ind[1]].x,
                                                             corner_points_less_sharp_ptr->points[point_search_ind[1]].y,
                                                             corner_points_less_sharp_ptr->points[point_search_ind[1]].z);

                                double s;
                                if (DISTORTION)
                                    s = (point_ori.intensity - int(point_ori.intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;

                                ceres::CostFunction *cost_function = LidarEdgeFactorLast::Create(curr_point, last_point_a, last_point_b, s);
                                //ceres::CostFunction *cost_function = new LidarEdgeFactor_z_rot_xy_trans(curr_point, last_point_a, last_point_b, s);
                                //ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, last_point_a);
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                corner_correspondence++;
                            } else if (change_ind == 1) {
                                if (point_search_sq_dis[4] < 0.01 * sqrt_dis) {
                                    Eigen::Vector3d center(0, 0, 0);
                                    for (int j = 0; j < 5; j++)
                                    {
                                        Eigen::Vector3d tmp(corner_points_less_sharp_ptr->points[point_search_ind[j]].x,
                                                            corner_points_less_sharp_ptr->points[point_search_ind[j]].y,
                                                            corner_points_less_sharp_ptr->points[point_search_ind[j]].z);
                                        center = center + tmp;
                                    }
                                    center = center / 5.0;
                                    Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                                    ceres::CostFunction *cost_function = LidarDistanceFactorLast::Create(curr_point, center);
                                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                }
                            } else {
                                int ind_temp = point_search_ind[change_ind];
                                float sq_dis_temp = point_search_sq_dis[change_ind];
                                point_search_ind[change_ind] = point_search_ind[1];
                                point_search_sq_dis[change_ind] = point_search_sq_dis[1];
                                point_search_ind[1] = ind_temp;
                                point_search_sq_dis[1] = sq_dis_temp;
                                if (point_search_sq_dis[4] < 0.01 * sqrt_dis) {
                                    Eigen::Vector3d center(0, 0, 0);
                                    for (int j = 0; j < 5; j++)
                                    {
                                        Eigen::Vector3d tmp(corner_points_less_sharp_ptr->points[point_search_ind[j]].x,
                                                            corner_points_less_sharp_ptr->points[point_search_ind[j]].y,
                                                            corner_points_less_sharp_ptr->points[point_search_ind[j]].z);
                                        center = center + tmp;
                                    }
                                    center = center / 5.0;
                                    Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                                    ceres::CostFunction *cost_function = LidarDistanceFactorLast::Create(curr_point, center);
                                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                }
                            }

                        }
                    }




                    if (using_last_flat_point == true)
                    {
                        PointCloud transformed_cloud;
                        pcl::transformPointCloud(*surface_points_flat_last_ptr, transformed_cloud, transf_se_.inverse());

                        for (int i = 0; i < surface_points_flat_last_num; ++i)
                        {


                            point_sel = transformed_cloud[i];
                            kdtree_surf_->nearestKSearch(point_sel, 6, point_search_ind, point_search_sq_dis);
                            if (point_search_sq_dis[0] < DISTANCE_SQ_THRESHOLD) {

                                int first_point_ring = int(surface_points_less_flat_ptr->points[point_search_ind[0]].intensity);
                                int second_point_ring = int(surface_points_less_flat_ptr->points[point_search_ind[1]].intensity);
                                int third_point_ring = int(surface_points_less_flat_ptr->points[point_search_ind[2]].intensity);

                                if (first_point_ring == second_point_ring && second_point_ring == third_point_ring) {
                                    for (int n = 3; n < 6; ++n) {
                                        int next_point_ring = int(surface_points_less_flat_ptr->points[point_search_ind[n]].intensity);
                                        if (third_point_ring != next_point_ring) {
                                            third_point_ring = next_point_ring;
                                            point_search_ind[2] = point_search_ind[n];
                                            point_search_sq_dis[2] = point_search_sq_dis[n];
                                            break;
                                        }
                                    }

                                    if (second_point_ring == third_point_ring) {
                                        continue;
                                    }
                                }
                                if (point_search_sq_dis[2] < DISTANCE_SQ_THRESHOLD) {
                                    Eigen::Vector3d curr_point(surface_points_flat_last_ptr->points[i].x,
                                                               surface_points_flat_last_ptr->points[i].y,
                                                               surface_points_flat_last_ptr->points[i].z);
                                    Eigen::Vector3d last_point_a(surface_points_less_flat_ptr->points[point_search_ind[0]].x,
                                                                 surface_points_less_flat_ptr->points[point_search_ind[0]].y,
                                                                 surface_points_less_flat_ptr->points[point_search_ind[0]].z);
                                    Eigen::Vector3d last_point_b(surface_points_less_flat_ptr->points[point_search_ind[1]].x,
                                                                 surface_points_less_flat_ptr->points[point_search_ind[1]].y,
                                                                 surface_points_less_flat_ptr->points[point_search_ind[1]].z);
                                    Eigen::Vector3d last_point_c(surface_points_less_flat_ptr->points[point_search_ind[2]].x,
                                                                 surface_points_less_flat_ptr->points[point_search_ind[2]].y,
                                                                 surface_points_less_flat_ptr->points[point_search_ind[2]].z);

                                    double s;
                                    if (DISTORTION)
                                        s = (surface_points_flat_last_ptr->points[i].intensity - int(surface_points_flat_last_ptr->points[i].intensity)) / SCAN_PERIOD;
                                    else
                                        s = 1.0;
                                    ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
// 				    ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
//                                     ceres::CostFunction *cost_function = new LidarPlaneFactor_z_rot_xy_trans(curr_point, last_point_a, last_point_b, last_point_c, s);
                                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                    plane_correspondence++;                                    
                                }
                            }
                        }
                    }


//                     if (using_full_point == true){
// 		      
// 		      PointCloud transformed_cloud;
// 		      pcl::transformPointCloud(*full_ponits_ptr, transformed_cloud, transf_se_);
// // 		      LOG(INFO) << full_ponits_num;
// 		      for(int i = 0; i < full_ponits_num; ++i){
// 			
// 			point_sel = transformed_cloud[i];
// 			int change_ind = 1;
// 			kdtree_full_->nearestKSearch(point_sel, 1, point_search_ind, point_search_sq_dis);
// 			
// 			Eigen::Vector3d curr_point(full_ponits_ptr->points[i].x,
// 			                           full_ponits_ptr->points[i].y,
// 			                           full_ponits_ptr->points[i].z);
// 			Eigen::Vector3d last_point(full_ponits_last_ptr->points[point_search_ind[0]].x,
// 			                           full_ponits_last_ptr->points[point_search_ind[0]].y,
// 			                           full_ponits_last_ptr->points[point_search_ind[0]].z);
// 			ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, last_point);
// 			problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
// 			
// 		     }
// 		      
// 		 }
                    
                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    printf("data association time %f ms \n", t_data.Toc());

                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        printf("less correspondence! *************************************************\n");
                    }

                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
//                     options.max_num_iterations = 4;
		    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                    options.minimizer_progress_to_stdout = true;
		    
		    options.use_nonmonotonic_steps = false;
		    options.gradient_tolerance = 1e-15;
		    options.function_tolerance = 1e-15;
		    
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
LOG(INFO) << summary.BriefReport();
                    transf_se_.linear() = q_last_curr.normalized().toRotationMatrix();
                    transf_se_.translation() = t_last_curr;                
                }
                
                
//                 for(size_t opti_counter = 0; opti_counter < 5; ++opti_counter){
// 		  
// 		  corner_correspondence = 0;
//                   plane_correspondence = 0;
// 		  
// 		  ceres::LossFunction *loss_function = new ceres::CauchyLoss(0.1);
//                   ceres::LocalParameterization *q_parameterization =
//                         new ceres::EigenQuaternionParameterization();
//                   ceres::Problem::Options problem_options;
// 		  
// 		  ceres::Problem problem(problem_options);
//                   problem.AddParameterBlock(para_q, 4, q_parameterization);
//                   problem.AddParameterBlock(para_t, 3);
// 		  PointT point_sel;
//                   std::vector<int> point_search_ind;
//                   std::vector<float> point_search_sq_dis;
// 		  
// 		  if (using_full_point == true){
// 		      
// 		      PointCloud transformed_cloud;
// 		      pcl::transformPointCloud(*full_ponits_ptr, transformed_cloud, transf_se_);
// // 		      LOG(INFO) << full_ponits_num;
// 		      for(int i = 0; i < full_ponits_num; ++i){
// 			
// 			point_sel = transformed_cloud[i];
// 			int change_ind = 1;
// 			kdtree_full_->nearestKSearch(point_sel, 1, point_search_ind, point_search_sq_dis);
// 			
// 			Eigen::Vector3d curr_point(full_ponits_ptr->points[i].x,
// 			                           full_ponits_ptr->points[i].y,
// 			                           full_ponits_ptr->points[i].z);
// 			Eigen::Vector3d last_point(full_ponits_last_ptr->points[point_search_ind[0]].x,
// 			                           full_ponits_last_ptr->points[point_search_ind[0]].y,
// 			                           full_ponits_last_ptr->points[point_search_ind[0]].z);
// 			ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, last_point);
// 			problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
// 			
// 		     }
// 		      
// 		 }
// 		 
// 		 ceres::Solver::Options options;
//                  options.use_nonmonotonic_steps = false;                      
//                  options.gradient_tolerance = 1e-15;
//                  options.function_tolerance = 1e-15;                        
//                  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
//                  options.linear_solver_type = ceres::DENSE_QR;
// 		 options.minimizer_progress_to_stdout = true;
//                  ceres::Solver::Summary summary;
//                  ceres::Solve(options, &problem, &summary);
// 		 LOG(INFO) << summary.BriefReport() << "OKOK111";
// 		 transf_se_.linear() = q_last_curr.normalized().toRotationMatrix();
//                  transf_se_.translation() = t_last_curr;
// 		  
// 	       }
                
                
                transf_sum_ = transf_sum_ * transf_se_;
                q_w_curr = Eigen::Quaterniond(transf_sum_.rotation());
                t_w_curr = transf_sum_.translation();     
	

                if (using_local_map == true) {




                    if (trans_sum_for_local_map.size() > 0) {


                        PointCloudPtr less_flat_mapping_point_ptr;
                        PointCloud less_flat_mapping_point;

                        PointCloudPtr less_sharp_mapping_point_ptr;
                        PointCloud less_sharp_mapping_point;

                        PointCloudPtr sharp_mapping_point_ptr;
                        PointCloud sharp_mapping_cloud;
                        PointCloud sharp_mapping_point;

                        if (using_flat_point == true){
                            for (int i = 0; i < local_map_less_flat.size(); ++i) {
                                PointCloud transformed_cloud;
                                pcl::transformPointCloud(local_map_less_flat[i], transformed_cloud, trans_sum_for_local_map[i]);
                                less_flat_mapping_point += transformed_cloud;
                            }

                            pcl::VoxelGrid<PointT> down_size_filter;
                            down_size_filter.setInputCloud(less_flat_mapping_point.makeShared());
                            down_size_filter.setLeafSize(0.2, 0.2, 0.2);

                            PointCloud less_flat_mapping_point_downsampled;
                            down_size_filter.filter(less_flat_mapping_point_downsampled);

                            less_flat_mapping_point_ptr = less_flat_mapping_point_downsampled.makeShared();

                            kdtree_surf_last_->setInputCloud(less_flat_mapping_point_ptr);

                            sensor_msgs::PointCloud2 local_mapping_flat;
                            pcl::toROSMsg(less_flat_mapping_point, local_mapping_flat);
                            local_mapping_flat.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                            local_mapping_flat.header.frame_id = "/laser_init";
                            pubLocalMappingLessFlat.publish(local_mapping_flat);
                        }


                        if (using_sharp_point == true) {


                            for (int i = 0; i < local_map_less_sharp.size(); ++i) {
                                PointCloud transformed_cloud;
                                pcl::transformPointCloud(local_map_less_sharp[i], transformed_cloud, trans_sum_for_local_map[i]);
                                less_sharp_mapping_point += transformed_cloud;
                            }
// LOG(INFO) <<  less_sharp_mapping_point.size();

                            pcl::VoxelGrid<PointT> down_size_filter;
                            down_size_filter.setInputCloud(less_sharp_mapping_point.makeShared());
                            down_size_filter.setLeafSize(0.2, 0.2, 0.2);

                            PointCloud less_sharp_mapping_point_downsampled;
                            down_size_filter.filter(less_sharp_mapping_point_downsampled);

                            less_sharp_mapping_point_ptr.reset(new PointCloud(less_sharp_mapping_point_downsampled));
// LOG(INFO) <<  less_sharp_mapping_point_ptr->size();
                            kdtree_corner_last_->setInputCloud(less_sharp_mapping_point_ptr);

                            sensor_msgs::PointCloud2 local_mapping_less_sharp;
                            pcl::toROSMsg(*less_sharp_mapping_point_ptr, local_mapping_less_sharp);
                            local_mapping_less_sharp.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                            local_mapping_less_sharp.header.frame_id = "/laser_init";
                            pubLocalMappingLessSharp.publish(local_mapping_less_sharp);
                        }




                        if (using_last_sharp_point == true) {
                            for (int i = 0; i < local_map_sharp.size(); ++i) {
                                PointCloud transformed_cloud;
                                sharp_mapping_cloud += local_map_sharp[i];
                                pcl::transformPointCloud(local_map_sharp[i], transformed_cloud, trans_sum_for_local_map[i]);
                                sharp_mapping_point += transformed_cloud;
                            }

                            pcl::VoxelGrid<PointT> down_size_filter;
                            down_size_filter.setInputCloud(sharp_mapping_point.makeShared());
                            down_size_filter.setLeafSize(0.2, 0.2, 0.2);

                            PointCloud sharp_mapping_point_downsampled;
                            down_size_filter.filter(sharp_mapping_point_downsampled);

                            sharp_mapping_point_ptr.reset(new PointCloud(sharp_mapping_point_downsampled));
                        }


                        for (size_t opti_counter = 0; opti_counter < 5; ++opti_counter)
                        {
                            corner_correspondence = 0;
                            plane_correspondence = 0;

                            //ceres::LossFunction *loss_function = NULL;
//                             ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                            ceres::LossFunction *loss_function =  new ceres::CauchyLoss(0.05);
                            ceres::LocalParameterization *q_parameterization =
                                new ceres::EigenQuaternionParameterization();
                            ceres::Problem::Options problem_options;

                            ceres::Problem problem(problem_options);
                            problem.AddParameterBlock(transf_sum_para_q, 4, q_parameterization);
                            problem.AddParameterBlock(transf_sum_para_t, 3);
                            PointT point_sel;
                            PointT point_norm;
                            PointT point_1;

                            std::vector<int> point_search_ind;
                            std::vector<float> point_search_sq_dis;

                            TicToc t_data;


                        // find correspondence for plane features

                            if (using_flat_point == true)
                            {
                                PointCloud transformed_cloud;
                                pcl::transformPointCloud(*surface_points_flat_ptr, transformed_cloud, transf_sum_);

                                for (int i = 0; i < surface_points_flat_num; ++i)
                                {
                                    point_sel = transformed_cloud[i];
                                    // TransformPoint(transf_sum_.rot, transf_sum_.pos, surfPointsFlatNorm->points[i], point_norm);

                                    // Eigen::Vector3d curr_point_norm(point_norm.x - transf_sum_.pos(0), 
                                    //                                 point_norm.y - transf_sum_.pos(1),
                                    //                                 point_norm.z - transf_sum_.pos(2));
                                    kdtree_surf_last_->nearestKSearch(point_sel, 6, point_search_ind, point_search_sq_dis); 
                                    if (point_search_sq_dis[0] < DISTANCE_SQ_THRESHOLD) {

                                        int first_point_ring = int(less_flat_mapping_point_ptr->points[point_search_ind[0]].intensity);
                                        int second_point_ring = int(less_flat_mapping_point_ptr->points[point_search_ind[1]].intensity);
                                        int third_point_ring = int(less_flat_mapping_point_ptr->points[point_search_ind[2]].intensity);

                                        if (first_point_ring == second_point_ring && second_point_ring == third_point_ring) {
                                            for (int n = 3; n < 6; ++n) {
                                                int next_point_ring = int(less_flat_mapping_point_ptr->points[point_search_ind[n]].intensity);
                                                if (third_point_ring != next_point_ring) {
                                                    third_point_ring = next_point_ring;
                                                    point_search_ind[2] = point_search_ind[n];
                                                    point_search_sq_dis[2] = point_search_sq_dis[n];
                                                    break;
                                                }
                                            }

                                            if (second_point_ring == third_point_ring) {
                                                continue;
                                            }
                                        }
                                        if (point_search_sq_dis[2] < DISTANCE_SQ_THRESHOLD) {
                                            Eigen::Vector3d curr_point(surface_points_flat_ptr->points[i].x,
                                                                       surface_points_flat_ptr->points[i].y,
                                                                       surface_points_flat_ptr->points[i].z);
                                            Eigen::Vector3d last_point_a(less_flat_mapping_point_ptr->points[point_search_ind[0]].x,
                                                                         less_flat_mapping_point_ptr->points[point_search_ind[0]].y,
                                                                         less_flat_mapping_point_ptr->points[point_search_ind[0]].z);
                                            Eigen::Vector3d last_point_b(less_flat_mapping_point_ptr->points[point_search_ind[1]].x,
                                                                         less_flat_mapping_point_ptr->points[point_search_ind[1]].y,
                                                                         less_flat_mapping_point_ptr->points[point_search_ind[1]].z);
                                            Eigen::Vector3d last_point_c(less_flat_mapping_point_ptr->points[point_search_ind[2]].x,
                                                                         less_flat_mapping_point_ptr->points[point_search_ind[2]].y,
                                                                         less_flat_mapping_point_ptr->points[point_search_ind[2]].z);

                                            double s;
                                            if (DISTORTION)
                                                s = (surface_points_flat_ptr->points[i].intensity - int(surface_points_flat_ptr->points[i].intensity)) / SCAN_PERIOD;
                                            else
                                                s = 1.0;
                                            ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                            //ceres::CostFunction *cost_function = LidarPlane_Norm_Factor::Create(curr_point, last_point_a, last_point_b, last_point_c, curr_point_norm, s);
                                            problem.AddResidualBlock(cost_function, loss_function, transf_sum_para_q, transf_sum_para_t);
                                            plane_correspondence++;                                    
                                        } else {
                                        continue;
                                        }

                                    } else {
                                        continue;
                                    }
                                }
                            }







                            if (using_sharp_point == true)
                            {
                                PointCloud transformed_cloud;
                                pcl::transformPointCloud(*corner_points_sharp_ptr, transformed_cloud, transf_sum_);

                                for (int i = 0; i < corner_points_sharp_num; ++i)
                                {
                                    point_sel = transformed_cloud[i];
                                    int change_ind = 1;
                                    PointT point_ori = corner_points_sharp_ptr->points[i];
                                    double sqrt_dis = point_ori.x * point_ori.x + point_ori.y * point_ori.y + point_ori.z * point_ori.z;

                                    kdtree_corner_last_->nearestKSearch(point_sel, 6, point_search_ind, point_search_sq_dis);
                                    int first_point_ring = int(less_sharp_mapping_point_ptr->points[point_search_ind[0]].intensity);
                                    int second_point_ring = int(less_sharp_mapping_point_ptr->points[point_search_ind[1]].intensity);

                                    if (first_point_ring == second_point_ring) {
                                        for (int n = 2; n < 6; ++n) {
                                            int next_point_ring = int(less_sharp_mapping_point_ptr->points[point_search_ind[n]].intensity);
                                            if (second_point_ring != next_point_ring) {
                                                second_point_ring = next_point_ring;
                                                int ind_temp = point_search_ind[n];
                                                float sq_dis_temp = point_search_sq_dis[n];
                                                point_search_ind[n] = point_search_ind[1];
                                                point_search_sq_dis[n] = point_search_sq_dis[1];
                                                point_search_ind[1] = ind_temp;
                                                point_search_sq_dis[1] = sq_dis_temp;
                                                change_ind = n;
                                                break;

                                            }
                                        }
                                    }
                                    if (first_point_ring != second_point_ring && point_search_sq_dis[1] < DISTANCE_SQ_THRESHOLD) {
                                        Eigen::Vector3d curr_point(corner_points_sharp_ptr->points[i].x,
                                                                corner_points_sharp_ptr->points[i].y,
                                                                corner_points_sharp_ptr->points[i].z);
                                        Eigen::Vector3d last_point_a(less_sharp_mapping_point_ptr->points[point_search_ind[0]].x,
                                                                    less_sharp_mapping_point_ptr->points[point_search_ind[0]].y,
                                                                    less_sharp_mapping_point_ptr->points[point_search_ind[0]].z);
                                        Eigen::Vector3d last_point_b(less_sharp_mapping_point_ptr->points[point_search_ind[1]].x,
                                                                    less_sharp_mapping_point_ptr->points[point_search_ind[1]].y,
                                                                    less_sharp_mapping_point_ptr->points[point_search_ind[1]].z);

                                        double s;
                                        if (DISTORTION)
                                            s = (corner_points_sharp_ptr->points[i].intensity - int(corner_points_sharp_ptr->points[i].intensity)) / SCAN_PERIOD;
                                        else
                                            s = 1.0;

                                        ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                                        //ceres::CostFunction *cost_function = new LidarEdgeFactor_z_rot_xy_trans(curr_point, last_point_a, last_point_b, s);
                                        //ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, last_point_a);
                                        problem.AddResidualBlock(cost_function, loss_function, transf_sum_para_q, transf_sum_para_t);
                                        corner_correspondence++;
                                    } else if (change_ind == 1) {
                                        if (point_search_sq_dis[4] < 0.01 * sqrt_dis) {
                                            Eigen::Vector3d center(0, 0, 0);
                                            for (int j = 0; j < 5; j++)
                                            {

                                                Eigen::Vector3d tmp(less_sharp_mapping_point_ptr->points[point_search_ind[j]].x,
                                                                    less_sharp_mapping_point_ptr->points[point_search_ind[j]].y,
                                                                    less_sharp_mapping_point_ptr->points[point_search_ind[j]].z);
                                                center = center + tmp;
                                            }
                                            center = center / 5.0;
                                            Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                                            ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
                                            problem.AddResidualBlock(cost_function, loss_function, transf_sum_para_q, transf_sum_para_t);
                                        }
                                    } else {
                                        int ind_temp = point_search_ind[change_ind];
                                        float sq_dis_temp = point_search_sq_dis[change_ind];
                                        point_search_ind[change_ind] = point_search_ind[1];
                                        point_search_sq_dis[change_ind] = point_search_sq_dis[1];
                                        point_search_ind[1] = ind_temp;
                                        point_search_sq_dis[1] = sq_dis_temp;
                                        if (point_search_sq_dis[4] < 0.01 * sqrt_dis) {
                                            Eigen::Vector3d center(0, 0, 0);
                                            for (int j = 0; j < 5; j++)
                                            {

                                                Eigen::Vector3d tmp(less_sharp_mapping_point_ptr->points[point_search_ind[j]].x,
                                                                    less_sharp_mapping_point_ptr->points[point_search_ind[j]].y,
                                                                    less_sharp_mapping_point_ptr->points[point_search_ind[j]].z);
                                                center = center + tmp;
                                        }
                                            center = center / 5.0;
                                            Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                                            ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
                                            problem.AddResidualBlock(cost_function, loss_function, transf_sum_para_q, transf_sum_para_t);
                                        }
                                    }

                                }
                            }





                            if (using_last_sharp_point == true)
                            {
                                PointCloud transformed_cloud;
                                pcl::transformPointCloud(sharp_mapping_point, transformed_cloud, transf_sum_.inverse());
                                size_t sharp_mapping_point_num = transformed_cloud.size();

                                for (int i = 0; i < sharp_mapping_point_num; ++i)
                                {
                                    int change_ind = 1;
                                    PointT point_ori = sharp_mapping_point[i];
                                    PointT point = sharp_mapping_cloud[i];

                                    double sqrt_dis = point.x * point.x + point.y * point.y + point.z * point.z;

                                    kdtree_corner_->nearestKSearch(transformed_cloud[i], 6, point_search_ind, point_search_sq_dis);

                                    int first_point_ring = int(corner_points_less_sharp_ptr->points[point_search_ind[0]].intensity);
                                    int second_point_ring = int(corner_points_less_sharp_ptr->points[point_search_ind[1]].intensity);

                                    if (first_point_ring == second_point_ring) {
                                        for (int n = 2; n < 6; ++n) {
                                            int next_point_ring = int(corner_points_less_sharp_ptr->points[point_search_ind[n]].intensity);
                                            if (second_point_ring != next_point_ring) {
                                                second_point_ring = next_point_ring;
                                                int ind_temp = point_search_ind[n];
                                                float sq_dis_temp = point_search_sq_dis[n];
                                                point_search_ind[n] = point_search_ind[1];
                                                point_search_sq_dis[n] = point_search_sq_dis[1];
                                                point_search_ind[1] = ind_temp;
                                                point_search_sq_dis[1] = sq_dis_temp;
                                                change_ind = n;
                                                break;

                                            }
                                        }
                                    }

                                    if (first_point_ring != second_point_ring && point_search_sq_dis[1] < DISTANCE_SQ_THRESHOLD) {
                                        Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                                        Eigen::Vector3d last_point_a(corner_points_less_sharp_ptr->points[point_search_ind[0]].x,
                                                                    corner_points_less_sharp_ptr->points[point_search_ind[0]].y,
                                                                    corner_points_less_sharp_ptr->points[point_search_ind[0]].z);
                                        Eigen::Vector3d last_point_b(corner_points_less_sharp_ptr->points[point_search_ind[1]].x,
                                                                    corner_points_less_sharp_ptr->points[point_search_ind[1]].y,
                                                                    corner_points_less_sharp_ptr->points[point_search_ind[1]].z);

                                        double s;
                                        if (DISTORTION)
                                            s = (corner_points_sharp_ptr->points[i].intensity - int(corner_points_sharp_ptr->points[i].intensity)) / SCAN_PERIOD;
                                        else
                                            s = 1.0;

                                        ceres::CostFunction *cost_function = LidarEdgeFactorLast::Create(curr_point, last_point_a, last_point_b, s);
                                        //ceres::CostFunction *cost_function = new LidarEdgeFactor_z_rot_xy_trans(curr_point, last_point_a, last_point_b, s);
                                        //ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, last_point_a);
                                        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                        corner_correspondence++;
                                    } else if (change_ind == 1) {
                                        if (point_search_sq_dis[4] < 0.01 * sqrt_dis) {
                                            Eigen::Vector3d center(0, 0, 0);
                                            for (int j = 0; j < 5; j++)
                                            {
                                                Eigen::Vector3d tmp(corner_points_less_sharp_ptr->points[point_search_ind[j]].x,
                                                                    corner_points_less_sharp_ptr->points[point_search_ind[j]].y,
                                                                    corner_points_less_sharp_ptr->points[point_search_ind[j]].z);
                                                center = center + tmp;
                                            }
                                            center = center / 5.0;
                                            Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                                            ceres::CostFunction *cost_function = LidarDistanceFactorLast::Create(curr_point, center);
                                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                        }
                                    } else {
                                        int ind_temp = point_search_ind[change_ind];
                                        float sq_dis_temp = point_search_sq_dis[change_ind];
                                        point_search_ind[change_ind] = point_search_ind[1];
                                        point_search_sq_dis[change_ind] = point_search_sq_dis[1];
                                        point_search_ind[1] = ind_temp;
                                        point_search_sq_dis[1] = sq_dis_temp;
                                        if (point_search_sq_dis[4] < 0.01 * sqrt_dis) {
                                            Eigen::Vector3d center(0, 0, 0);
                                            for (int j = 0; j < 5; j++)
                                            {
                                                Eigen::Vector3d tmp(corner_points_less_sharp_ptr->points[point_search_ind[j]].x,
                                                                    corner_points_less_sharp_ptr->points[point_search_ind[j]].y,
                                                                    corner_points_less_sharp_ptr->points[point_search_ind[j]].z);
                                                center = center + tmp;
                                            }
                                            center = center / 5.0;
                                            Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                                            ceres::CostFunction *cost_function = LidarDistanceFactorLast::Create(curr_point, center);
                                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                        }
                                    }






                                    // PointT point_ori = corner_points_sharp_ptr->points[i];
                                    // double sqrt_dis = point_ori.x * point_ori.x + point_ori.y * point_ori.y + point_ori.z * point_ori.z;

                                    // kdtree_corner_last_->nearestKSearch(point_sel, 5, point_search_ind, point_search_sq_dis); 

                                    // if (point_search_sq_dis[4] < DISTANCE_SQ_THRESHOLD)
                                    // {
                                    //     std::vector<Eigen::Vector3d> nearCorners;
                                    //     Eigen::Vector3d center(0, 0, 0);
                                    //     for (int j = 0; j < 5; j++)
                                    //     {
                                    //         Eigen::Vector3d tmp(corner_points_less_sharp_last_ptr->points[point_search_ind[j]].x,
                                    //                             corner_points_less_sharp_last_ptr->points[point_search_ind[j]].y,
                                    //                             corner_points_less_sharp_last_ptr->points[point_search_ind[j]].z);
                                    //         center = center + tmp;
                                    //         nearCorners.push_back(tmp);
                                    //     }
                                    //     center = center / 5.0;

                                    //     Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                                    //     for (int j = 0; j < 5; j++)
                                    //     {
                                    //         Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                                    //         covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                                    //     }

                                    //     Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                                    //     // if is indeed line feature
                                    //     // note Eigen library sort eigenvalues in increasing order
                                    //     Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                                    //     Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                                    //     if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                                    //     {
                                    //         Eigen::Vector3d point_on_line = center;
                                    //         Eigen::Vector3d point_a, point_b;
                                    //         point_a = 0.1 * unit_direction + point_on_line;
                                    //         point_b = -0.1 * unit_direction + point_on_line;

                                    //         ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
                                    //         //ceres::CostFunction *cost_function = new LidarEdgeFactor_z_rot_xy_trans(curr_point, point_a, point_b, 1.0);
                                    //         problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                    //         corner_correspondence++; 
                                    //     }							
                                    // } else if (point_search_sq_dis[4] < 0.01 * sqrt_dis) {
                                    //     Eigen::Vector3d center(0, 0, 0);
                                    //     for (int j = 0; j < 5; j++)
                                    //     {
                                    //         Eigen::Vector3d tmp(corner_points_less_sharp_last_ptr->points[point_search_ind[j]].x,
                                    //                             corner_points_less_sharp_last_ptr->points[point_search_ind[j]].y,
                                    //                             corner_points_less_sharp_last_ptr->points[point_search_ind[j]].z);
                                    //         center = center + tmp;
                                    //     }
                                    //     center = center / 5.0;
                                    //     Eigen::Vector3d curr_point(point_ori.x, point_ori.y, point_ori.z);
                                    //     ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
                                    //     problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                    // }








                                    // kdtree_corner_last_->nearestKSearch(point_sel, 1, point_search_ind, point_search_sq_dis);
                                    // int closest_point_ind = -1, min_point_ind2 = -1;
                                    // if (point_search_sq_dis[0] < DISTANCE_SQ_THRESHOLD)
                                    // {
                                    //     closest_point_ind = point_search_ind[0];
                                    //     int closest_point_scan_id = int(corner_points_less_sharp_last_ptr->points[closest_point_ind].intensity);

                                    //     double min_point_sq_dis2 = DISTANCE_SQ_THRESHOLD;
                                    //     // search in the direction of increasing scan line
                                    //     for (int j = closest_point_ind + 1; j < (int)corner_points_less_sharp_last_ptr->points.size(); ++j)
                                    //     {
                                    //         // if in the same scan line, continue
                                    //         if (int(corner_points_less_sharp_last_ptr->points[j].intensity) <= closest_point_scan_id)
                                    //             continue;

                                    //         // if not in nearby scans, end the loop
                                    //         if (int(corner_points_less_sharp_last_ptr->points[j].intensity) > (closest_point_scan_id + NEARBY_SCAN))
                                    //             break;

                                    //         double point_sq_dis = (corner_points_less_sharp_last_ptr->points[j].x - point_sel.x) *
                                    //                                 (corner_points_less_sharp_last_ptr->points[j].x - point_sel.x) +
                                    //                             (corner_points_less_sharp_last_ptr->points[j].y - point_sel.y) *
                                    //                                 (corner_points_less_sharp_last_ptr->points[j].y - point_sel.y) +
                                    //                             (corner_points_less_sharp_last_ptr->points[j].z - point_sel.z) *
                                    //                                 (corner_points_less_sharp_last_ptr->points[j].z - point_sel.z);

                                    //         if (point_sq_dis < min_point_sq_dis2)
                                    //         {
                                    //             // find nearer point
                                    //             min_point_sq_dis2 = point_sq_dis;
                                    //             min_point_ind2 = j;
                                    //         }
                                    //     }

                                    //     // search in the direction of decreasing scan line
                                    //     for (int j = closest_point_ind - 1; j >= 0; --j)
                                    //     {
                                    //         // if in the same scan line, continue
                                    //         if (int(corner_points_less_sharp_last_ptr->points[j].intensity) >= closest_point_scan_id)
                                    //             continue;

                                    //         // if not in nearby scans, end the loop
                                    //         if (int(corner_points_less_sharp_last_ptr->points[j].intensity) < (closest_point_scan_id - NEARBY_SCAN))
                                    //             break;

                                    //         double point_sq_dis = (corner_points_less_sharp_last_ptr->points[j].x - point_sel.x) *
                                    //                                 (corner_points_less_sharp_last_ptr->points[j].x - point_sel.x) +
                                    //                             (corner_points_less_sharp_last_ptr->points[j].y - point_sel.y) *
                                    //                                 (corner_points_less_sharp_last_ptr->points[j].y - point_sel.y) +
                                    //                             (corner_points_less_sharp_last_ptr->points[j].z - point_sel.z) *
                                    //                                 (corner_points_less_sharp_last_ptr->points[j].z - point_sel.z);

                                    //         if (point_sq_dis < min_point_sq_dis2)
                                    //         {
                                    //             // find nearer point
                                    //             min_point_sq_dis2 = point_sq_dis;
                                    //             min_point_ind2 = j;
                                    //         }
                                    //     }
                                    // }

                                    // if (min_point_ind2 >= 0) // both closest_point_ind and min_point_ind2 is valid
                                    // {
                                    //     Eigen::Vector3d curr_point(corner_points_sharp_ptr->points[i].x,
                                    //                             corner_points_sharp_ptr->points[i].y,
                                    //                             corner_points_sharp_ptr->points[i].z);
                                    //     Eigen::Vector3d last_point_a(corner_points_less_sharp_last_ptr->points[closest_point_ind].x,
                                    //                                 corner_points_less_sharp_last_ptr->points[closest_point_ind].y,
                                    //                                 corner_points_less_sharp_last_ptr->points[closest_point_ind].z);
                                    //     Eigen::Vector3d last_point_b(corner_points_less_sharp_last_ptr->points[min_point_ind2].x,
                                    //                                 corner_points_less_sharp_last_ptr->points[min_point_ind2].y,
                                    //                                 corner_points_less_sharp_last_ptr->points[min_point_ind2].z);

                                    //     double s;
                                    //     if (DISTORTION)
                                    //         s = (corner_points_sharp_ptr->points[i].intensity - int(corner_points_sharp_ptr->points[i].intensity)) / SCAN_PERIOD;
                                    //     else
                                    //         s = 1.0;

                                    //     ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                                    //     //ceres::CostFunction *cost_function = new LidarEdgeFactor_z_rot_xy_trans(curr_point, last_point_a, last_point_b, s);
                                    //     problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                    //     corner_correspondence++;
                                    // }




                                }
                             }








                            printf("data association time %f ms \n", t_data.Toc());

                            if ((corner_correspondence + plane_correspondence) < 10)
                            {
                                printf("less correspondence! *************************************************\n");
                            }

                            TicToc t_solver;
                            ceres::Solver::Options options;
                            options.linear_solver_type = ceres::DENSE_QR;
//                             options.max_num_iterations = 4;
			    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                            options.minimizer_progress_to_stdout = true;
			    
			    options.use_nonmonotonic_steps = false;
		            options.gradient_tolerance = 1e-15;
		            options.function_tolerance = 1e-15;
			    
                            ceres::Solver::Summary summary;
                            ceres::Solve(options, &problem, &summary);
LOG(INFO) << summary.BriefReport();
                            transf_sum_.linear() = q_w_curr.normalized().toRotationMatrix();
                            transf_sum_.translation() = t_w_curr;    
//LOG(INFO) << summary.BriefReport() << std::endl;

                        }
                    }

                    transf_se_ = transf_sum_last.inverse() * transf_sum_;

                    q_last_curr = Eigen::Quaterniond(transf_se_.rotation());
                    t_last_curr = transf_se_.translation();

                    int key_fream_num = trans_sum_for_local_map.size();
                    Transform last_key_fream_trans = trans_sum_for_local_map[key_fream_num - 1];    

                    float delta_rot_keyfream = mathutils::RadToDeg(q_w_curr.angularDistance(last_key_fream_trans.rot));

                    Eigen::Vector3d trans_pre_curr = t_w_curr - last_key_fream_trans.pos;

                    float delta_trans_keyfream = trans_pre_curr.norm();

                    if (delta_trans_keyfream > 2) {
                        key_fream = true;
                    }




                }

            }

            transf_sum_last = transf_sum_;


            TicToc t_pub;

            // publish odometry
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/laser_init";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);

            //publish current to last odometry
            nav_msgs::Odometry laserOdometry_last_curr;
            laserOdometry_last_curr.header.frame_id = "/laser_init";
            laserOdometry_last_curr.child_frame_id = "/laser_odom";
            laserOdometry_last_curr.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            laserOdometry_last_curr.pose.pose.orientation.x = Eigen::Quaterniond(transf_se_.rotation()).x();
            laserOdometry_last_curr.pose.pose.orientation.y = Eigen::Quaterniond(transf_se_.rotation()).y();
            laserOdometry_last_curr.pose.pose.orientation.z = Eigen::Quaterniond(transf_se_.rotation()).z();
            laserOdometry_last_curr.pose.pose.orientation.w = Eigen::Quaterniond(transf_se_.rotation()).w();
            laserOdometry_last_curr.pose.pose.position.x = transf_se_.translation().x();
            laserOdometry_last_curr.pose.pose.position.y = transf_se_.translation().y();
            laserOdometry_last_curr.pose.pose.position.z = transf_se_.translation().z();
            pubLaserOdometry_last_curr.publish(laserOdometry_last_curr);
            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/laser_init";
            pubLaserPath.publish(laserPath);

            // transform corner features and plane features to the scan end point
            if (DISTORTION)
            {
                int cornerPointsLessSharpNum = corner_points_less_sharp_ptr->points.size();
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    TransformToEnd(&corner_points_less_sharp_ptr->points[i], &corner_points_less_sharp_ptr->points[i]);
                }

                int surfPointsLessFlatNum = surface_points_less_flat_ptr->points.size();
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    TransformToEnd(&surface_points_less_flat_ptr->points[i], &surface_points_less_flat_ptr->points[i]);
                }

                int laserCloudFullResNum = full_ponits_ptr->points.size();
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&full_ponits_ptr->points[i], &full_ponits_ptr->points[i]);
                }
            }
// LOG(INFO) << corner_points_less_sharp_last_ptr->size();
            corner_points_less_sharp_last_ptr = corner_points_less_sharp_ptr;
            corner_points_less_sharp_ptr.reset(new PointCloud());

            surface_points_less_flat_last_ptr = surface_points_less_flat_ptr;
            surface_points_less_flat_ptr.reset(new PointCloud());

// LOG(INFO) << corner_points_sharp_last_ptr->size();
            corner_points_sharp_last_ptr = corner_points_sharp_ptr;
            corner_points_sharp_ptr.reset(new PointCloud());

            surface_points_flat_last_ptr = surface_points_flat_ptr;
            surface_points_flat_ptr.reset(new PointCloud());

            full_ponits_last_ptr = full_ponits_ptr;
            full_ponits_ptr.reset(new PointCloud());
            
            kdtree_corner_last_->setInputCloud(corner_points_less_sharp_last_ptr);
            kdtree_surf_last_->setInputCloud(surface_points_less_flat_last_ptr);

	    kdtree_full_last_->setInputCloud(full_ponits_last_ptr);

            if (using_local_map == true) {
               
                if (first_fream || key_fream) {
                    local_map_flat.push(*surface_points_flat_last_ptr);
                    local_map_sharp.push(*corner_points_sharp_last_ptr);
                    local_map_less_flat.push(*surface_points_less_flat_last_ptr);
                    local_map_less_sharp.push(*corner_points_less_sharp_last_ptr);
                    trans_sum_for_local_map.push(transf_sum_);
                    first_fream = false;
                    key_fream = false;

                    pubKeyFreamOdometry.publish(laserOdometry);

                    sensor_msgs::PointCloud2 key_fream_cloud_less_flat;
                    pcl::toROSMsg(*surface_points_less_flat_last_ptr, key_fream_cloud_less_flat);
                    key_fream_cloud_less_flat.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                    key_fream_cloud_less_flat.header.frame_id = "/laser_init";
                    pubKeyFreamPointsLessFlat.publish(key_fream_cloud_less_flat);

                    sensor_msgs::PointCloud2 key_fream_cloud_less_sharp;
                    pcl::toROSMsg(*corner_points_less_sharp_last_ptr, key_fream_cloud_less_sharp);
                    key_fream_cloud_less_sharp.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                    key_fream_cloud_less_sharp.header.frame_id = "/laser_init";
                    pubKeyFreamPointsLessSharp.publish(key_fream_cloud_less_sharp);

                    sensor_msgs::PointCloud2 key_fream_cloud_flat;
                    pcl::toROSMsg(*surface_points_flat_ptr, key_fream_cloud_flat);
                    key_fream_cloud_flat.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                    key_fream_cloud_flat.header.frame_id = "/laser_init";
                    pubKeyFreamPointsFlat.publish(key_fream_cloud_flat);

                    sensor_msgs::PointCloud2 key_fream_cloud_sharp;
                    pcl::toROSMsg(*corner_points_sharp_ptr, key_fream_cloud_sharp);
                    key_fream_cloud_sharp.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                    key_fream_cloud_sharp.header.frame_id = "/laser_init";
                    pubKeyFreamPointsSharp.publish(key_fream_cloud_sharp);
                } else {
                    PointCloud less_sharp_cloud;
                    PointCloud less_flat_cloud;
                    size_t num = trans_sum_for_local_map.size();
                    Transform transf_sum_key = trans_sum_for_local_map[num - 1];
                    Transform transf_key_current = transf_sum_key.inverse() * transf_sum_;
                    pcl::transformPointCloud(*corner_points_less_sharp_last_ptr, less_sharp_cloud, transf_key_current.transform());
                    pcl::transformPointCloud(*surface_points_less_flat_last_ptr, less_flat_cloud, transf_key_current.transform());
                    local_map_less_sharp[num - 1] += less_sharp_cloud;
                    local_map_less_flat[num - 1] += less_flat_cloud;
                }
            }

            Sophus::SE3<double> transf_sum_SE3(Eigen::Quaterniond(transf_sum_.rotation()), transf_sum_.translation());

	    Eigen::Quaterniond q_sum = Eigen::Quaterniond(transf_sum_.rotation());
	    Eigen::Vector3d t_sum = transf_sum_.translation();
            Eigen::Matrix<double, 3, 4>  transf_sum_mat = transf_sum_SE3.matrix3x4();


	    std::ofstream result_out("/home/wzw/lom/test_odom.txt", std::ios::binary | std::ios::app | std::ios::out);
            result_out << std::setprecision(12) << std::scientific 
                                               << q_sum.w() << " " << q_sum.x() << " " << q_sum.y() << " " << q_sum.z() << " "
                                               << t_sum.x() << " " << - t_sum.y() << " " << t_sum.z() << std::endl;
            //result_out << std::flush;
            result_out.close();
	    
//             std::ofstream result_out("/home/wzw/lom/test_odom.txt", std::ios::binary | std::ios::app | std::ios::out);
//             result_out << std::setprecision(12) << std::scientific 
//                                                << transf_sum_mat(0,0) << " " << transf_sum_mat(0,1) << " " << transf_sum_mat(0,2) << " " << -transf_sum_mat(1,3) << " "
//                                                << transf_sum_mat(1,0) << " " << transf_sum_mat(1,1) << " " << transf_sum_mat(1,2) << " " << -transf_sum_mat(2,3) << " "
//                                                << transf_sum_mat(2,0) << " " << transf_sum_mat(2,1) << " " << transf_sum_mat(2,2) << " " << transf_sum_mat(0,3) << std::endl;
//             //result_out << std::flush;
//             result_out.close();


            if (frameCount % skipFrameNum == 0)
            {
                frameCount = 0;

                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*corner_points_less_sharp_last_ptr, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudCornerLast2.header.frame_id = "/laser";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*surface_points_less_flat_last_ptr, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudSurfLast2.header.frame_id = "/laser";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

                // sensor_msgs::PointCloud2 laserCloudSurfLast2;
                // pcl::toROSMsg(*surfPointsLessFlatMap, laserCloudSurfLast2);
                // laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                // laserCloudSurfLast2.header.frame_id = "/laser";
                // pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*full_ponits_ptr, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudFullRes3.header.frame_id = "/laser";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }
            printf("publication time %f ms \n", t_pub.Toc());
            printf("whole laserOdometry time %f ms \n \n", t_whole.Toc());
            if(t_whole.Toc() > 100)
                ROS_WARN("odometry process over 100ms");

            frameCount++;
        }
        rate.sleep();
    }
    return 0;
}
