#include <gtest/gtest.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/io/pcd_io.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_datatypes.h>

#include <geometry_msgs/Quaternion.h>

#include "../include/point_processor/PointProcessor.h"
#include "../include/utils/TicToc.h"

using namespace lom;
using namespace std;
using namespace mathutils;


int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  ros::init(argc, argv, "point_processor");

  ros::NodeHandle nh("~");

  PointProcessorConfig config;
  config.setup_param(nh);

  PointProcessor processor(config);

  LOG(INFO) << "sensor_type: " << processor.all_laser_scans_.size();
  LOG(INFO) << "using_rings_num: " << processor.laser_scans_.size();

  processor.SetupRos(nh);

  ros::Rate r(100);
  while (ros::ok()) {
    ros::spinOnce();
    r.sleep();
  }

  return 0;
}