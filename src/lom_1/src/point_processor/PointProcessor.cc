#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "../../include/point_processor/PointProcessor.h"

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>

//#define DEBUG
//#define DEBUG_ORIGIN

namespace lom {

double AbsRadDistance(double a, double b) {
  return fabs(NormalizeRad(a - b));
}

void PointProcessorConfig::setup_param(ros::NodeHandle &nh){
    nh.param("deskew", deskew, false);
    nh.param("even", even, true);
    nh.param("using_livox", using_livox, false);
    nh.param("sensor_type", sensor_type, 16);
    nh.param("deg_diff", deg_diff, 0.2);
    nh.param("scan_period", scan_period, 0.1);
    nh.param("lower_bound", lower_bound, -15.0);
    nh.param("upper_bound", upper_bound, 15.0);
    nh.param("using_lower_ring_num", using_lower_ring_num, 1);
    nh.param("using_upper_ring_num", using_upper_ring_num, 16);
    nh.param("num_scan_subregions", num_scan_subregions, 8);
    nh.param("num_curvature_regions_corner", num_curvature_regions_corner, 5);
    nh.param("num_curvature_regions_flat", num_curvature_regions_flat, 5);
    nh.param("num_feature_regions", num_feature_regions, 5);
    nh.param("surf_curv_th", surf_curv_th, 0.01);
    nh.param("sharp_curv_th", sharp_curv_th, 0.1);
    nh.param("max_corner_sharp", max_corner_sharp, 2);
    nh.param("max_corner_less_sharp", max_corner_less_sharp, 20);
    nh.param("max_surf_flat", max_surf_flat, 4);
    nh.param("max_surf_normal", max_surf_nomal, 30);
    nh.param("max_sq_dis", max_sq_dis, 1.0);

    nh.param("flat_extract_num", flat_extract_num, 100);
    nh.param("flat_extract_num_x_trans", flat_extract_num_x_trans, 100);
    nh.param("flat_extract_num_y_trans", flat_extract_num_y_trans, 100);
    nh.param("flat_extract_num_z_trans", flat_extract_num_z_trans, 100);
    nh.param("flat_extract_num_x_rot", flat_extract_num_x_rot, 100);
    nh.param("flat_extract_num_y_rot", flat_extract_num_y_rot, 100);
    nh.param("flat_extract_num_z_rot", flat_extract_num_z_rot, 100);

    nh.param("lower_ring_num_sharp_point", lower_ring_num_sharp_point, 6);
    nh.param("upper_ring_num_sharp_point", upper_ring_num_sharp_point, 16);

    nh.param("lower_ring_num_z_trans", lower_ring_num_z_trans, 1);
    nh.param("upper_ring_num_z_trans", upper_ring_num_z_trans, 6);
    
    nh.param("lower_ring_num_x_rot", lower_ring_num_x_rot, 1);
    nh.param("upper_ring_num_x_rot", upper_ring_num_x_rot, 6);
    nh.param("lower_ring_num_y_rot", lower_ring_num_y_rot, 1);
    nh.param("upper_ring_num_y_rot", upper_ring_num_y_rot, 6);
    nh.param("lower_ring_num_z_rot_xy_trans", lower_ring_num_z_rot_xy_trans, 6);
    nh.param("upper_ring_num_z_rot_xy_trans", upper_ring_num_z_rot_xy_trans, 16);

    nh.param("less_flat_filter_size", less_flat_filter_size, 0.2);
    nh.param("infer_start_ori", infer_start_ori, false);
    nh.param<string>("capture_frame_id", capture_frame_id, "/map");
    nh.param("using_surf_point_normal", using_surf_point_normal, true);
    nh.param("using_surf_point_2", using_surf_point_2, true);
    nh.param("extract_per_ring", extract_per_ring, true);
}

PointProcessor::PointProcessor(const PointProcessorConfig &config)
    : config_(config){
  time_factor_ = 1/config_.scan_period;
  factor_ = ((config_.sensor_type -1) / (config_.upper_bound - config_.lower_bound));
  if (config_.using_livox) {
    row_num_ = config_.using_upper_ring_num - config_.using_lower_ring_num + 1;
    col_num_ = 2 * int(config_.sensor_type * 81.7 / 25.1);
  } else {
    row_num_ = config_.using_upper_ring_num - config_.using_lower_ring_num + 1;
    col_num_ = 360.0/config_.deg_diff;
  }

  all_laser_scans_.clear();
  for (int i = 0; i < config_.sensor_type; ++i) {
    PointCloudPtr scan(new PointCloud());
    all_laser_scans_.push_back(scan);
  }    

  all_intensity_scans_.clear();
  for (int i = 0; i < config_.sensor_type; ++i) {
    PointCloudPtr scan(new PointCloud());
    all_intensity_scans_.push_back(scan);
  }

  laser_scans_.clear();
  for (int i = 0; i < row_num_; ++i) {
    PointCloudPtr scan(new PointCloud());
    laser_scans_.push_back(scan);
  }

  intensity_scans_.clear();
  for (int i = 0; i < row_num_; ++i) {
    PointCloudPtr scan(new PointCloud());
    intensity_scans_.push_back(scan);
  }

  range_image_.clear();
  for (int i = 0; i < row_num_; ++i) {
    std::vector<ImageElement> image_row(col_num_);
    range_image_.push_back(image_row);
  }
}

void PointProcessor::Process() {

  PointToRing();

  ExtractFeaturePoints();

  //ExtractFeaturePoints_2();

  PublishResults();

}

void PointProcessor::PointCloudHandler(const sensor_msgs::PointCloud2ConstPtr &raw_points_msg) {

  PointCloud laser_cloud_in;
  pcl::fromROSMsg(*raw_points_msg, laser_cloud_in);
  vector<int> ind;
  pcl::removeNaNFromPointCloud(laser_cloud_in, laser_cloud_in, ind);

  PointCloudConstPtr laser_cloud_in_ptr(new PointCloud(laser_cloud_in));

  SetInputCloud(laser_cloud_in_ptr, raw_points_msg->header.stamp);

  Process();
}

void PointProcessor::PointCloudHandler(const sensor_msgs::PointCloudConstPtr &raw_points_msg) {

  pcl::PointCloud<PointIR> laser_cloud_in;

  size_t point_num = raw_points_msg->points.size();
  for (int i = 0; i < point_num; ++i) {
  PointIR p_tmp;
  p_tmp.x = raw_points_msg->points[i].x;
  p_tmp.y = raw_points_msg->points[i].y;
  p_tmp.z = raw_points_msg->points[i].z;

  p_tmp.ring = raw_points_msg->channels[1].values[i];
  p_tmp.intensity = raw_points_msg->channels[0].values[i];
  laser_cloud_in.push_back(p_tmp);
  }

  pcl::PointCloud<PointIR>::Ptr laser_cloud_in_ptr(new pcl::PointCloud<PointIR>(laser_cloud_in));

  SetInputCloud(laser_cloud_in_ptr, raw_points_msg->header.stamp);
  Process();
}

void PointProcessor::TansfHandler(const nav_msgs::OdometryConstPtr &transf_se_msg) {
  transf_se_time_ = transf_se_msg->header.stamp;

  transf_se_pos_(0) = transf_se_msg->pose.pose.position.x;
  transf_se_pos_(1) = transf_se_msg->pose.pose.position.y;
  transf_se_pos_(2) = transf_se_msg->pose.pose.position.z;

  transf_se_rot_.x() = transf_se_msg->pose.pose.orientation.x;
  transf_se_rot_.y() = transf_se_msg->pose.pose.orientation.y;  
  transf_se_rot_.z() = transf_se_msg->pose.pose.orientation.z;
  transf_se_rot_.w() = transf_se_msg->pose.pose.orientation.w;
}

void PointProcessor::SetupRos(ros::NodeHandle &nh) {
  is_ros_setup_ = true;
  // subscribe to raw cloud topic
  if (config_.using_livox) {
    sub_raw_points_ = nh.subscribe<sensor_msgs::PointCloud2>
    ("/livox/lidar", 100, &PointProcessor::PointCloudHandler, this);
  } else if (config_.even) {
    sub_raw_points_ = nh.subscribe<sensor_msgs::PointCloud2>
    ("/laser_diy", 100, &PointProcessor::PointCloudHandler, this);
  } else {
    sub_raw_points_ = nh.subscribe<sensor_msgs::PointCloud>
    ("/lidar_test_points", 100, &PointProcessor::PointCloudHandler, this);    
  }
  sub_tansf_es_ = nh.subscribe<nav_msgs::Odometry>
      ("/laser_odom_to_last", 100, &PointProcessor::TansfHandler, this);

  // advertise scan registration topics
  pub_full_cloud_ = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud", 2);
  pub_corner_points_sharp_ = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 2);
  pub_corner_points_less_sharp_ = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 2);

  pub_surf_points_flat_ = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 2);
  pub_surf_points_less_flat_ = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 2);

  pub_surf_points_flat_z_trans_ = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat_z_trans", 2);
  pub_surf_points_flat_z_rot_xy_trans_ = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat_z_rot_xy_trans", 2);
  pub_surf_points_flat_x_rot_ = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat_x_rot", 2);
  pub_surf_points_flat_y_rot_ = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat_y_rot", 2);

  pub_surf_points_normal_ = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat_normal", 2);

  pub_start_ori_ = nh.advertise<std_msgs::Float32>("/debug/start_ori", 10);
  pub_start_ori_inferred_ = nh.advertise<std_msgs::Float32>("/debug/start_ori_inferred", 10);
}

void PointProcessor::Reset(const ros::Time &scan_time, const bool &is_new_sweep) {
  scan_time_ = scan_time;

  // clear internal cloud buffers at the beginning of a sweep
  if (is_new_sweep) {
    sweep_start_ = scan_time_;

    // clear cloud buffers
    cloud_in_rings_.clear();
    corner_points_sharp_.clear();
    corner_points_less_sharp_.clear();
    surface_points_flat_.clear();
    surface_points_less_flat_.clear();
    surface_points_flat_z_trans_.clear();
    surface_points_flat_z_rot_xy_trans_.clear();
    surface_points_flat_x_rot_.clear();
    surface_points_flat_y_rot_.clear();
    surface_points_normal_.clear();

    // clear scan indices vector
    scan_ranges_.clear();

    for (int i = 0; i < laser_scans_.size(); ++i) {
      laser_scans_[i]->clear();
    }

    for (int i = 0; i < intensity_scans_.size(); ++i) {
      intensity_scans_[i]->clear();
    }

    for (int i = 0; i < all_laser_scans_.size(); ++i) {
      all_laser_scans_[i]->clear();
    }

    for (int i = 0; i < all_intensity_scans_.size(); ++i) {
      all_intensity_scans_[i]->clear();
    }

  range_image_.clear();
  for (int i = 0; i < row_num_; ++i) {
    std::vector<ImageElement> image_row(col_num_);
    range_image_.push_back(image_row);
  }

//    laser_scans_.clear();
//
//    for (int i = 0; i < config_.sensor_type; ++i) {
//      PointCloudPtr scan(new PointCloud());
//      laser_scans_.push_back(scan);
//    }
  }
}

void PointProcessor::SetInputCloud(const PointCloudConstPtr &all_cloud_in, ros::Time time_in) {
  Reset(time_in);
  all_cloud_ptr_ = all_cloud_in;
}

void PointProcessor::SetInputCloud(const pcl::PointCloud<PointIR>::Ptr &all_cloud_in, ros::Time time_in) {
  Reset(time_in);
  all_cloud_ir_ptr_ = all_cloud_in;
}

void PointProcessor::PointToRing() {
  if (config_.using_livox) {
    PointToRing(all_cloud_ptr_, all_laser_scans_);
  } else if (config_.even) {
    PointToRing(all_cloud_ptr_, all_laser_scans_, all_intensity_scans_);
  } else {
    PointToRing(all_cloud_ir_ptr_, all_laser_scans_, all_intensity_scans_);
  }

  size_t cloud_size = 0;
  for (int i = 0; i < row_num_; i++) {
    cloud_in_rings_ += (*laser_scans_[i]);

    IndexRange range(cloud_size, 0);
    cloud_size += (*laser_scans_[i]).size();
    range.second = (cloud_size > 0 ? cloud_size - 1 : 0);
    scan_ranges_.push_back(range);
  }
// LOG(INFO) << cloud_in_rings_.size();
// LOG(INFO) << "OK";
  if (config_.deskew == true) {
      DeSkew(cloud_in_rings_);
  }
}




void PointProcessor::PointToRing(const PointCloudConstPtr &all_cloud_in,
                                 vector<PointCloudPtr> &all_ring,
                                 vector<PointCloudPtr> &all_intensity) {
  auto &points = all_cloud_in->points;
  size_t cloud_size = points.size();

  float startOri = -atan2(points[0].y, points[0].x);
  float endOri = -atan2(points[cloud_size - 1].y,
                        points[cloud_size - 1].x) + 2 * M_PI;

  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }

  bool halfPassed = false;
  bool start_flag = false;

  for (size_t i = 0; i < cloud_size; ++i) {
    PointT p = points[i];
    PointT p_with_intensity = points[i];
    float distance = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);

    if (distance > 300 || distance < 5) {
      continue;
    }

    float dis = sqrt(p.x * p.x + p.y * p.y);
    float ele_rad = atan2(p.z, dis);
    float azi_rad = 2 * M_PI - atan2(p.y, p.x);

    ///>mine
#ifndef DEBUG_ORIGIN
    if (azi_rad >= 2 * M_PI) {
      azi_rad -= 2 * M_PI;//azi_rad∈[0,360)
    }

    int scan_id = ElevationToRing(ele_rad);//scan_id∈[0,sensor_type-1]
    if (scan_id >= config_.sensor_type || scan_id < 0 || scan_id >= config_.sensor_type || config_.upper_bound < RadToDeg(ele_rad) || RadToDeg(ele_rad) < config_.lower_bound) {
      // DLOG(INFO) << RadToDeg(ele_rad) << ", " << scan_id;
      // DLOG(INFO) << (scan_id < 0 ? " point too low" : "point too high");
      continue;
    }

    if (!start_flag) {
      start_ori_ = azi_rad;
      start_flag = true;
    }

    // cout << scan_id << " " << azi_rad << endl;

//    if (config_.deskew) {
//      p.intensity = azi_rad;
//    } else {
//      p.intensity = scan_id;
//    }

    p.intensity = azi_rad;
#endif
    ///<mine

    ///>origin
#ifdef DEBUG_ORIGIN
    int scan_id = ElevationToRing(ele_rad);

    if (scan_id >= config_.sensor_type || scan_id < 0) {
      // DLOG(INFO) << RadToDeg(ele_rad) << ", " << scan_id;
      // DLOG(INFO) << (scan_id < 0 ? " point too low" : "point too high");
      continue;
    }

    float ori = -atan2(p.y, p.x);
    if (!halfPassed) {
      if (ori < startOri - M_PI / 2) {
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }

      if (ori - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      ori += 2 * M_PI;

      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      }
    }

    float relTime = (ori - startOri) / (endOri - startOri);
#ifndef DEBUG
    p.intensity = scan_id + config_.scan_period * relTime;
#else
    p.intensity = config_.scan_period * relTime;
#endif
#endif
    ///<origin
    all_ring[scan_id]->push_back(p);
    all_intensity[scan_id]->push_back(p_with_intensity);

  }

  ROS_DEBUG_STREAM("ring time: " << tic_toc_.Toc() << " ms");

  tic_toc_.Tic();
  // TODO: De-skew with rel_time, this for loop is not necessary
  // start_ori_ = NAN;

//  for (int ring = 0; ring < config_.sensor_type; ++ring) {
//    if (all_ring[ring]->size() <= 0) {
//      continue;
//    }
//
//    float azi_rad = all_ring[ring]->front().intensity;
//    if (start_ori_ != start_ori_) {
//      start_ori_ = azi_rad;
//    } else {
//      start_ori_ = (RadLt(start_ori_, azi_rad) ? start_ori_ : azi_rad);
//    }
//    // cout << azi_rad << endl;
//  }
  // start_ori_ = all_ring[0]->front().intensity;

  // infer right start_ori
  if (config_.infer_start_ori) {
    std_msgs::Float32 start_ori_msg, start_ori_inferred_msg;
    start_ori_msg.data = start_ori_;

    start_ori_buf2_.push(start_ori_);
    if (start_ori_buf1_.size() >= 10) {
      float start_ori_diff1 = start_ori_buf1_.last() - start_ori_buf1_.first();
      float start_ori_step1 = NormalizeRad(start_ori_diff1) / 9;

      float start_ori_diff2 = start_ori_buf2_.last() - start_ori_buf2_.first();
      float start_ori_step2 = NormalizeRad(start_ori_diff2) / 9;

      DLOG(INFO) << "origin start_ori_: " << start_ori_;
      DLOG(INFO) << "start_ori_step1: " << start_ori_step1;
      DLOG(INFO) << "start_ori_step2: " << start_ori_step2;
//       DLOG(INFO) << "diff: " << fabs(NormalizeRad(start_ori_ - start_ori_buf_.last()));
      if (fabs(NormalizeRad(start_ori_ - start_ori_buf1_.last())) > config_.deg_diff) {
        start_ori_ = start_ori_buf1_.last() + start_ori_step1;
        start_ori_ = NormalizeRad(start_ori_);
        if (start_ori_ < 0) {
          start_ori_ += 2 * M_PI;
        }
      }
      if (AbsRadDistance(start_ori_step1, start_ori_step2) < 0.05
          && AbsRadDistance(start_ori_buf2_[9] - start_ori_buf2_[8], start_ori_step1) < 0.05
          && AbsRadDistance(start_ori_buf2_[8] - start_ori_buf2_[7], start_ori_step1) < 0.05
          && AbsRadDistance(start_ori_buf2_[7] - start_ori_buf2_[6], start_ori_step1) < 0.05
          && AbsRadDistance(start_ori_buf2_[6] - start_ori_buf2_[5], start_ori_step1) < 0.05
          && AbsRadDistance(start_ori_buf2_[5] - start_ori_buf2_[4], start_ori_step1) < 0.05
          && AbsRadDistance(start_ori_buf2_[4] - start_ori_buf2_[3], start_ori_step1) < 0.05
          && AbsRadDistance(start_ori_buf2_[3] - start_ori_buf2_[2], start_ori_step1) < 0.05
          && AbsRadDistance(start_ori_buf2_[2] - start_ori_buf2_[1], start_ori_step1) < 0.05
          && AbsRadDistance(start_ori_buf2_[1] - start_ori_buf2_[0], start_ori_step1) < 0.05) {
        start_ori_ = all_ring[0]->front().intensity;
      }
    }
    start_ori_buf1_.push(start_ori_);

    start_ori_inferred_msg.data = start_ori_;
    pub_start_ori_.publish(start_ori_msg);
    pub_start_ori_inferred_.publish(start_ori_inferred_msg);
  }  // if

//DLOG(INFO) << "start_ori_: " << start_ori_;

  for (int ring = 0; ring < config_.sensor_type; ++ring) {
    // points in a ring
    PointCloud &points_in_ring = (*all_ring[ring]);
    PointCloud &points_in_ring_with_intensity = (*all_intensity[ring]);
    size_t cloud_in_ring_size = points_in_ring.size();

    for (int i = 0; i < cloud_in_ring_size; ++i) {
      PointT &p = points_in_ring[i];
      PointT &p_with_intensity = points_in_ring_with_intensity[i];

      float azi_rad_rel = p.intensity - start_ori_;
      if (azi_rad_rel < 0) {
        azi_rad_rel += 2 * M_PI;
      }

      float rel_time = config_.scan_period * azi_rad_rel / (2 * M_PI);
      ///>mine
#ifndef DEBUG_ORIGIN

#ifndef DEBUG
      p.intensity = ring + rel_time;
      p_with_intensity.intensity = int(p_with_intensity.intensity) + rel_time;
#else
      p.intensity = rel_time;
#endif

#endif

    int row = ring - config_.using_lower_ring_num + 1;
    int col = int(azi_rad_rel / (2 * M_PI) * col_num_);

    if (row >= 0 && row < row_num_ && col >= 0 && col < col_num_) {

      if (range_image_[row][col].occupy_state == 0) {
        range_image_[row][col].point = p;
        range_image_[row][col].occupy_state = 1;
      } else {
        float point_dis_1 = CalcPointDistance(p);
        float point_dis_2 = CalcPointDistance(range_image_[row][col].point);
        if (point_dis_1 < point_dis_2) {
          range_image_[row][col].point = p;
        }
      }
    }

      ///<mine
    }  // for i
  }  // for ring
  ROS_DEBUG_STREAM("reorder time: " << tic_toc_.Toc() << " ms");

  for (int i = 0; i < row_num_; ++i) {
    laser_scans_[i] = all_ring[i + config_.using_lower_ring_num - 1];
    sort (laser_scans_[i]->points.begin(), laser_scans_[i]->points.end(),
      [](const PointT& p_1, const PointT& p_2){
        return p_1.intensity < p_2.intensity;
      });    
    intensity_scans_[i] = all_intensity[i + config_.using_lower_ring_num - 1];
// LOG(INFO) << i << " " << laser_scans_[i]->size();
  }

// double last_aa = 0.0;
// for (int i = 0; i < 64; ++i) {
//   double aa = 0.0;
//   int a = laser_scans_[i]->size();
//   for (int j = 0; j < a; ++j) {
//     auto& point = laser_scans_[i]->points[j];

//     float dis = sqrt(point.x * point.x + point.y * point.y);
//     float ele_rad = RadToDeg(atan2(point.z, dis));
//     aa += ele_rad;
//   }
//   aa = aa/a;

// LOG(INFO) << aa - last_aa << " " << i << " " << a << " " << aa;

//   last_aa = aa;
// }





//LOG(INFO) << "I ok";
}




void PointProcessor::PointToRing(const PointCloudConstPtr &all_cloud_in,
                                 vector<PointCloudPtr> &all_ring) {
  auto &points = all_cloud_in->points;
  size_t cloud_size = points.size();

int a = 0;
  for (size_t i = 0; i < cloud_size; ++i) {
    PointT p = points[i];
    PointT p_with_intensity = points[i];
    float distance = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);

    // if (distance > 300 || distance < 5) {
    //   continue;
    // }

    if (distance < 0.01) {
      ++a;
      continue;
    }

    double ring_factor = config_.sensor_type / 25.1;

    float dis_xy = sqrt(p.x * p.x + p.y * p.y);
    float ele_deg = RadToDeg(atan2(p.z, dis_xy));
    int scan_id = int((ele_deg + 25.1 /2) *  ring_factor + 0.5);
    if (scan_id >= config_.sensor_type || scan_id < 0) {
      continue;
    }

    float dis_xz = sqrt(p.x * p.x + p.z * p.z);
    float horizontal_deg = RadToDeg(atan2(p.y, dis_xz));
    float relative_hotizontal = (horizontal_deg + 81.7 /2) / 81.7;

    p.intensity = scan_id + relative_hotizontal;
    all_ring[scan_id]->push_back(p);

    int row = scan_id - config_.using_lower_ring_num + 1;
    int col = int((horizontal_deg + 81.7 /2) / 81.7 * col_num_ + 0.5);

    if (row >= 0 && row < row_num_ && col >= 0 && col < col_num_) {

      if (range_image_[row][col].occupy_state == 0) {
        range_image_[row][col].point = p;
        range_image_[row][col].occupy_state = 1;
      } else {
        float point_dis_1 = CalcPointDistance(p);
        float point_dis_2 = CalcPointDistance(range_image_[row][col].point);
        if (point_dis_1 < point_dis_2) {
          range_image_[row][col].point = p;
        }
      }
    }
  }

  for(int i = 0; i < row_num_; ++i){
    laser_scans_[i] = all_ring[i + config_.using_lower_ring_num - 1];
    sort (laser_scans_[i]->points.begin(), laser_scans_[i]->points.end(),
      [](const PointT& p_1, const PointT& p_2){
        return p_1.intensity < p_2.intensity;
      });
  }
}


void PointProcessor::PointToRing(const pcl::PointCloud<PointIR>::Ptr &all_cloud_in,
                                 vector<PointCloudPtr> &all_ring,
                                 vector<PointCloudPtr> &all_intensity) {
  auto &points = all_cloud_in->points;
  size_t cloud_size = points.size();

  float startOri = -atan2(points[0].y, points[0].x);
  float endOri = -atan2(points[cloud_size - 1].y,
                        points[cloud_size - 1].x) + 2 * M_PI;
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }


  bool start_flag = false;
 
  for (size_t i = 0; i < cloud_size; ++i) {
    PointT p;
    p.x = points[i].x;
    p.y = points[i].y;
    p.z = points[i].z;
    p.intensity = points[i].intensity;
   
    PointT p_with_intensity = p;

    float azi_rad = 2 * M_PI - atan2(p.y, p.x);

    if (azi_rad >= 2 * M_PI) {
      azi_rad -= 2 * M_PI;
    }

    int scan_id = points[i].ring;

    if (scan_id >= config_.sensor_type || scan_id < 0) {
      continue;
    }

    if (!start_flag) {
      start_ori_ = azi_rad;
      start_flag = true;
    }

    float azi_rad_rel = azi_rad - start_ori_;

    if (azi_rad_rel < 0) {
      azi_rad_rel += 2 * M_PI;
    }

    p.intensity = scan_id + config_.scan_period * azi_rad_rel / (2 * M_PI);
    all_ring[scan_id]->push_back(p);
    all_intensity[scan_id]->push_back(p_with_intensity);

    int row = scan_id - config_.using_lower_ring_num + 1;
    int col = int(azi_rad_rel / (2 * M_PI) * col_num_);
    if (row >= 0 && row < row_num_ && col >= 0 && col < col_num_) {
      if (range_image_[row][col].occupy_state == 0) {
        range_image_[row][col].point = p;
        range_image_[row][col].occupy_state = 1;
      } else {
        float point_dis_1 = CalcPointDistance(p);
        float point_dis_2 = CalcPointDistance(range_image_[row][col].point);
        if (point_dis_1 < point_dis_2) {
          range_image_[row][col].point = p;
        }
      }
    }
  }



// for (int i = 0; i < row_num_; ++i) {
//   int a = 0;
//   for (int j = 0; j < col_num_; ++j) {
//     if (range_image_[i][j].occupy_state == 1) {
// LOG(INFO) << i << " " << j;
//       if (j > a + 5) {
//         corner_points_sharp_.push_back(range_image_[i][a].point);
//         corner_points_sharp_.push_back(range_image_[i][j].point);
//       }
//       a = j;
//     }
//   }
// }

  for(int i = 0; i < row_num_; ++i){
    laser_scans_[i] = all_ring[i + config_.using_lower_ring_num - 1];
    sort (laser_scans_[i]->points.begin(), laser_scans_[i]->points.end(),
      [](const PointT& p_1, const PointT& p_2){
        return p_1.intensity < p_2.intensity;
      });
    intensity_scans_[i] = all_intensity[i + config_.using_lower_ring_num - 1];
  }

}


void PointProcessor::DeSkew(PointCloud &points) {
  size_t points_num = points.size();

  if (float((sweep_start_ - transf_se_time_).toSec()) < 2 * (config_.scan_period + 0.01)) {
    for(int i =0; i < points_num; ++i) {
      Eigen::Vector3f point(points[i].x, points[i].y, points[i].z);
      float s = time_factor_ * (points[i].intensity - int(points[i].intensity));
      if (s < 0 || s > 1.001) {
        LOG(ERROR) << "time ratio error: " << s;
        return;
      }

      Eigen::Quaternionf transf_sp_rot;
      Eigen::Quaternionf quat_temp{1.0, 0.0, 0.0, 0.0};
      transf_sp_rot = quat_temp.slerp(s, transf_se_rot_).normalized();
      point = transf_sp_rot * point + s * transf_se_pos_;

      points[i].x = point(0);
      points[i].y = point(1);
      points[i].z = point(2);
    }  
  }
}



void PointProcessor::PrepareRing_corner(const PointCloud &scan) {

  size_t scan_size = scan.size();
  scan_ring_mask_.resize(scan_size);
  scan_ring_mask_.assign(scan_size, 0);
  for (size_t i = 0 + config_.num_curvature_regions_corner; i < scan_size - config_.num_curvature_regions_corner; ++i) {
    const PointT &p_prev = scan[i - 1];
    const PointT &p_curr = scan[i];
    const PointT &p_next = scan[i + 1];

    float diff_next2 = CalcSquaredDiff(p_curr, p_next);

    // about 30 cm
    if (diff_next2 > 0.1) {
      float depth = CalcPointDistance(p_curr);
      float depth_next = CalcPointDistance(p_next);

      if (depth > depth_next) {
        // to closer point
        float weighted_diff = sqrt(CalcSquaredDiff(p_next, p_curr, depth_next / depth)) / depth_next;
        // relative distance
        if (weighted_diff < 0.1) {
          fill_n(&scan_ring_mask_[i - 0 - config_.num_curvature_regions_corner], config_.num_curvature_regions_corner, 1);
          continue;
        }
      } else {
        float weighted_diff = sqrt(CalcSquaredDiff(p_curr, p_next, depth / depth_next)) / depth;
        if (weighted_diff < 0.1) {
          fill_n(&scan_ring_mask_[i - 0 + 1], config_.num_curvature_regions_corner - 1, 1);
          continue;
        }
      }
    }

    float diff_prev2 = CalcSquaredDiff(p_curr, p_prev);
    float dis2 = CalcSquaredPointDistance(p_curr);

    // for this point -- 1m -- 1.5cm
    if (diff_next2 > 0.0002 * dis2 && diff_prev2 > 0.0002 * dis2) {
      scan_ring_mask_[i - 0] = 1;
    }

  }
}


void PointProcessor::PrepareRing_flat(const PointCloud &scan) {

  size_t scan_size = scan.size();
  // scan_ring_mask_.resize(scan_size);
  // scan_ring_mask_.assign(scan_size, 0);
  for (size_t i = 0 + config_.num_curvature_regions_flat; i < scan_size - config_.num_curvature_regions_flat; ++i) {
    const PointT &p_prev = scan[i - 1];
    const PointT &p_curr = scan[i];
    const PointT &p_next = scan[i + 1];

    float diff_next2 = CalcSquaredDiff(p_curr, p_next);

    // about 30 cm
    if (diff_next2 > 0.1) {
      float depth = CalcPointDistance(p_curr);
      float depth_next = CalcPointDistance(p_next);

      if (depth > depth_next) {
        // to closer point
        float weighted_diff = sqrt(CalcSquaredDiff(p_next, p_curr, depth_next / depth)) / depth_next;
        // relative distance
        if (weighted_diff < 0.1) {
          fill_n(&scan_ring_mask_[i - 0 - config_.num_curvature_regions_flat], config_.num_curvature_regions_flat, 1);
          continue;
        }
      } else {
        float weighted_diff = sqrt(CalcSquaredDiff(p_curr, p_next, depth / depth_next)) / depth;
        if (weighted_diff < 0.1) {
          fill_n(&scan_ring_mask_[i - 0 + 1], config_.num_curvature_regions_flat - 1, 1);
          continue;
        }
      }
    }

    // float diff_prev2 = CalcSquaredDiff(p_curr, p_prev);
    // float dis2 = CalcSquaredPointDistance(p_curr);

    // // for this point -- 1m -- 1.5cm
    // if (diff_next2 > 0.0002 * dis2 && diff_prev2 > 0.0002 * dis2) {
    //   scan_ring_mask_[i - 0] = 1;
    // }

  }
}

void PointProcessor::PrepareSubregion_corner(const PointCloud &scan, const size_t idx_start, const size_t idx_end) {

//  cout << ">>>>>>> " << idx_ring << ", " << idx_start << ", " << idx_end << " <<<<<<<" << endl;
//  const PointCloud &scan = laser_scans_[idx_ring];
  size_t region_size = idx_end - idx_start + 1;
  curvature_idx_pairs_.resize(region_size);
  subregion_labels_.resize(region_size);
  subregion_labels_.assign(region_size, SURFACE_LESS_FLAT);


  for (size_t i = idx_start, in_region_idx = 0; i <= idx_end; ++i, ++in_region_idx) {


    int num_point_neighbors = 2 * config_.num_curvature_regions_corner;
    float diff_x = -num_point_neighbors * scan[i].x;
    float diff_y = -num_point_neighbors * scan[i].y;
    float diff_z = -num_point_neighbors * scan[i].z;

    for (int j = 1; j <= config_.num_curvature_regions_corner; j++) {
      diff_x += scan[i + j].x + scan[i - j].x;
      diff_y += scan[i + j].y + scan[i - j].y;
      diff_z += scan[i + j].z + scan[i - j].z;
    }


    // float point_dist = CalcPointDistance(scan[i]);
    // int num_curvature_regions = int(10 / point_dist) + 1;

    // int num_point_neighbors = 2 * num_curvature_regions;
    // float diff_x = -num_point_neighbors * scan[i].x;
    // float diff_y = -num_point_neighbors * scan[i].y;
    // float diff_z = -num_point_neighbors * scan[i].z;

    // if (i - num_curvature_regions < 0) {
    //   num_curvature_regions = config_.num_curvature_regions;
    //   for (int j = 1; j <= num_curvature_regions; j++) {
    //     diff_x += scan[i + j].x + scan[i - j].x;
    //     diff_y += scan[i + j].y + scan[i - j].y;
    //     diff_z += scan[i + j].z + scan[i - j].z;
    //   }
    // } else {
    //   for (int j = 1; j <= num_curvature_regions; j++) {
    //     diff_x += scan[i + j].x + scan[i - j].x;
    //     diff_y += scan[i + j].y + scan[i - j].y;
    //     diff_z += scan[i + j].z + scan[i - j].z;
    //   }
    // }


    float curvature = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
    pair<float, size_t> curvature_idx_(curvature, i);
    curvature_idx_pairs_[in_region_idx] = curvature_idx_;
//    _regionCurvature[regionIdx] = diffX * diffX + diffY * diffY + diffZ * diffZ;
//    _regionSortIndices[regionIdx] = i;
  }

  sort(curvature_idx_pairs_.begin(), curvature_idx_pairs_.end());
/*  
  for (const auto &pair : curvature_idx_pairs_) {
    cout << pair.first << " " << pair.second << endl;
  }
*/
}

void PointProcessor::PrepareSubregion_flat(const PointCloud &scan, const size_t idx_start, const size_t idx_end) {

//  cout << ">>>>>>> " << idx_ring << ", " << idx_start << ", " << idx_end << " <<<<<<<" << endl;
//  const PointCloud &scan = laser_scans_[idx_ring];
  size_t region_size = idx_end - idx_start + 1;
  size_t scan_size = scan.size();
  curvature_idx_pairs_.resize(region_size);
  subregion_labels_.resize(region_size);
  subregion_labels_.assign(region_size, SURFACE_LESS_FLAT);

  for (size_t i = idx_start, in_region_idx = 0; i <= idx_end; ++i, ++in_region_idx) {

    // int num_point_neighbors = 2 * config_.num_curvature_regions_flat;
    // float diff_x = -num_point_neighbors * scan[i].x;
    // float diff_y = -num_point_neighbors * scan[i].y;
    // float diff_z = -num_point_neighbors * scan[i].z;

    // for (int j = 1; j <= config_.num_curvature_regions_flat; j++) {
    //   diff_x += scan[i + j].x + scan[i - j].x;
    //   diff_y += scan[i + j].y + scan[i - j].y;
    //   diff_z += scan[i + j].z + scan[i - j].z;
    // }


    float point_dist = CalcPointDistance(scan[i]);
    int num_curvature_regions = int(20 / point_dist) + 1;

    if (i - num_curvature_regions < 0 || i + num_curvature_regions > scan_size - 1) {
      num_curvature_regions = config_.num_curvature_regions_flat;
    }

    int num_point_neighbors = 2 * num_curvature_regions;
    float diff_x = -num_point_neighbors * scan[i].x;
    float diff_y = -num_point_neighbors * scan[i].y;
    float diff_z = -num_point_neighbors * scan[i].z;


    for (int j = 1; j <= num_curvature_regions; j++) {
      diff_x += scan[i + j].x + scan[i - j].x;
      diff_y += scan[i + j].y + scan[i - j].y;
      diff_z += scan[i + j].z + scan[i - j].z;
    }

    float curvature = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
    pair<float, size_t> curvature_idx_(curvature, i);
    curvature_idx_pairs_[in_region_idx] = curvature_idx_;
//    _regionCurvature[regionIdx] = diffX * diffX + diffY * diffY + diffZ * diffZ;
//    _regionSortIndices[regionIdx] = i;
  }
  sort(curvature_idx_pairs_.begin(), curvature_idx_pairs_.end());
/*  
  for (const auto &pair : curvature_idx_pairs_) {
    cout << pair.first << " " << pair.second << endl;
  }
*/
}

void PointProcessor::MaskPickedInRing(const PointCloud &scan, const size_t in_scan_idx) {

  // const PointCloud &scan = laser_scans_[idx_ring];
  scan_ring_mask_[in_scan_idx] = 1;

  for (int i = 1; i <= config_.num_feature_regions; ++i) {
    /// 20cm
    if (CalcSquaredDiff(scan[in_scan_idx + i], scan[in_scan_idx + i - 1]) > 0.04) {
      break;
    }

    scan_ring_mask_[in_scan_idx + i] = 1;
  }

  for (int i = 1; i <= config_.num_feature_regions; ++i) {
    if (CalcSquaredDiff(scan[in_scan_idx - i], scan[in_scan_idx - i + 1]) > 0.04) {
      break;
    }

    scan_ring_mask_[in_scan_idx - i] = 1;
  }
}


PointCloud::iterator PointProcessor::find_binary(const PointCloud::iterator &begin, const PointCloud::iterator &end, const float &intensity){
  PointCloud::iterator l = begin;
  PointCloud::iterator r = end - 1;
  PointCloud::iterator mid;
  while (l <= r) {
    mid = l + distance(l, r)/2;
    if (mid->intensity > intensity) {
      r = mid - 1;
    }
    else if (intensity > mid->intensity) {
      l = mid + 1;
    }
    else 
      return mid;
  }
  return l;
}





void PointProcessor::ExtractFeaturePoints() {
  tic_toc_.Tic();

  vector<PointCloud> surface_points_flat_temp_vec(row_num_);
  vector<PointCloud> surface_points_flat_temp_vec_deskew(row_num_);

  // surface_points_flat_temp_vec.clear();
  // surface_points_flat_temp_vec_deskew.clear(); 
  ///< i is #ring, j is #subregion, k is # in region
  for (size_t i = 0; i < row_num_; ++i) {

    size_t start_idx = scan_ranges_[i].first;
    size_t end_idx = scan_ranges_[i].second;

    // skip too short scans
    if (config_.num_curvature_regions_corner < config_.num_curvature_regions_flat) {
      if (end_idx <= start_idx + 2 * config_.num_curvature_regions_flat) {
        continue;
      }      
    } else {
      if (end_idx <= start_idx + 2 * config_.num_curvature_regions_corner) {
        continue;
      }      
    }

    const PointCloud &scan_ring = *laser_scans_[i];
    size_t scan_size = scan_ring.size();

    PrepareRing_corner(scan_ring);
    //extract features from equally sized scan regions
    for (int j = 0; j < config_.num_scan_subregions; ++j) {
      // ((s+d)*N+j*(e-s-d))/N, ((s+d)*N+(j+1)*(e-s-d))/N-1
      size_t sp = ((0 + config_.num_curvature_regions_corner) * (config_.num_scan_subregions - j)
          + (scan_size - config_.num_curvature_regions_corner) * j) / config_.num_scan_subregions;
      size_t ep = ((0 + config_.num_curvature_regions_corner) * (config_.num_scan_subregions - 1 - j)
          + (scan_size - config_.num_curvature_regions_corner) * (j + 1)) / config_.num_scan_subregions - 1;

      // skip empty regions
      if (ep <= sp) {
        continue;
      }

      size_t region_size = ep - sp + 1;

      // extract corner features
      PrepareSubregion_corner(scan_ring, sp, ep);
      int num_largest_picked = 0;
      for (size_t k = region_size; k > 0 && num_largest_picked < config_.max_corner_less_sharp; --k) {

        // k must be greater than 0
        const pair<float, size_t> &curvature_idx = curvature_idx_pairs_[k-1];
        float curvature = curvature_idx.first;
        size_t idx = curvature_idx.second;
        size_t in_scan_idx = idx - 0; // scan start index is 0 for all ring scans
        size_t in_region_idx = idx - sp;

        if (scan_ring[in_scan_idx].z < 0) {
          continue;
        }

        int ring_inf = int(scan_ring[in_scan_idx].intensity);
        if (config_.lower_ring_num_sharp_point <= (ring_inf + 1) && (ring_inf + 1) <= config_.upper_ring_num_sharp_point) {

          if (scan_ring_mask_[in_scan_idx] == 0 && curvature > config_.sharp_curv_th) {

            if (in_scan_idx - config_.num_curvature_regions_corner >= 0 && in_scan_idx + 1 <= scan_size - config_.num_curvature_regions_corner) {
              PointT p_curr = scan_ring[in_scan_idx];
              PointT p_prev = scan_ring[in_scan_idx - config_.num_curvature_regions_corner];
              PointT p_next = scan_ring[in_scan_idx + config_.num_curvature_regions_corner];
              Eigen::Vector3f p_curr_vec(p_curr.x, p_curr.y, p_curr.z);
              Eigen::Vector3f line_prev(p_curr.x - p_prev.x, p_curr.y - p_prev.y, p_curr.z - p_prev.z);
              Eigen::Vector3f line_next(p_curr.x - p_next.x, p_curr.y - p_next.y, p_curr.z - p_next.z);
              p_curr_vec.normalize();
              line_prev.normalize();
              line_next.normalize();
              if (fabs(p_curr_vec.dot(line_prev)) > 0.985 && fabs(p_curr_vec.dot(line_next)) > 0.985) {
                continue;
              }
            }

            ++num_largest_picked;

            if (num_largest_picked <= config_.max_corner_sharp) {
              subregion_labels_[in_region_idx] = CORNER_SHARP;
              corner_points_sharp_.push_back(scan_ring[in_scan_idx]);
            } else {
              subregion_labels_[in_region_idx] = CORNER_LESS_SHARP;
            }
            corner_points_less_sharp_.push_back(scan_ring[in_scan_idx]);

            MaskPickedInRing(scan_ring, in_scan_idx);
          }
        }
      }
    } /// j
    PrepareRing_flat(scan_ring);
    // extract features from equally sized scan regions
    for (int j = 0; j < config_.num_scan_subregions; ++j) {
      // ((s+d)*N+j*(e-s-d))/N, ((s+d)*N+(j+1)*(e-s-d))/N-1
      size_t sp = ((0 + config_.num_curvature_regions_flat) * (config_.num_scan_subregions - j)
          + (scan_size - config_.num_curvature_regions_flat) * j) / config_.num_scan_subregions;
      size_t ep = ((0 + config_.num_curvature_regions_flat) * (config_.num_scan_subregions - 1 - j)
          + (scan_size - config_.num_curvature_regions_flat) * (j + 1)) / config_.num_scan_subregions - 1;

      // skip empty regions
      if (ep <= sp) {
        continue;
      }

      size_t region_size = ep - sp + 1;

      


      // extract flat surface features
      PrepareSubregion_flat(scan_ring, sp, ep);
      if (config_.using_surf_point_normal == true) {
        int num_smallest_picked = 0;
        for (int k = 0; k < region_size; ++k) {
          const pair<float, size_t> &curvature_idx = curvature_idx_pairs_[k];
          float curvature = curvature_idx.first;

          size_t idx = curvature_idx.second;
          size_t in_scan_idx = idx - 0; // scan start index is 0 for all ring scans
          size_t in_region_idx = idx - sp;
          if (curvature < config_.surf_curv_th && scan_ring_mask_[in_scan_idx] == 0) {
            ++num_smallest_picked;
            subregion_labels_[in_region_idx] = SURFACE_LESS_FLAT;
            surface_points_less_flat_.push_back(scan_ring[in_scan_idx]);
            surface_points_flat_temp_vec[i].push_back(scan_ring[in_scan_idx]);
            surface_points_flat_temp_vec_deskew[i].push_back(scan_ring[in_scan_idx]);
            MaskPickedInRing(scan_ring, in_scan_idx);
          }
        }
      } else {
        int num_smallest_picked = 0;
        for (int k = 0; k < region_size && num_smallest_picked < config_.max_surf_flat; ++k) {
          const pair<float, size_t> &curvature_idx = curvature_idx_pairs_[k];
          float curvature = curvature_idx.first;

          size_t idx = curvature_idx.second;
          size_t in_scan_idx = idx - 0; // scan start index is 0 for all ring scans
          size_t in_region_idx = idx - sp;

          if (scan_ring_mask_[in_scan_idx] == 0 && curvature < config_.surf_curv_th) {
            ++num_smallest_picked;
            subregion_labels_[in_region_idx] = SURFACE_LESS_FLAT;
            surface_points_flat_.push_back(scan_ring[in_scan_idx]);

            MaskPickedInRing(scan_ring, in_scan_idx);
          }
        }
      }

      if (!config_.using_surf_point_normal) {
        for (int k = 0; k < region_size; ++k) {
          if (curvature_idx_pairs_[k].first < config_.surf_curv_th) {
            surface_points_less_flat_.push_back(scan_ring[sp + k]);
          }
        }        
      }


    } /// j
  } /// i
  


  sort (corner_points_less_sharp_.begin(), corner_points_less_sharp_.end(),
    [](const PointT& p_1, const PointT& p_2){
      return p_1.intensity <= p_2.intensity;
    });

  if (config_.using_surf_point_normal) {
    sort (surface_points_less_flat_.begin(), surface_points_less_flat_.end(),
      [](const PointT& p_1, const PointT& p_2){
        return p_1.intensity <= p_2.intensity;
      });
  } else {
    pcl::VoxelGrid<PointT> down_size_filter;
    down_size_filter.setInputCloud(surface_points_less_flat_.makeShared());
    down_size_filter.setLeafSize(config_.less_flat_filter_size,
                                config_.less_flat_filter_size,
                                config_.less_flat_filter_size);
    down_size_filter.filter(surface_points_less_flat_);
    sort (surface_points_less_flat_.begin(), surface_points_less_flat_.end(),
      [](const PointT& p_1, const PointT& p_2){
        return p_1.intensity < p_2.intensity;
      });
  }



  if (config_.using_surf_point_normal) {

    pair<pair<PointT, PointT>, PointT> pair_nomal;

    size_t surf_points_flat_num = surface_points_less_flat_.size();
    PointCloud search_points;

    vector<pair<float, PointT>> dis_vector;
    pair<float, PointT> dis_pair;


    vector<pair<pair<PointT, PointT>, PointT>> vector_nomal_z_trans;
    vector<pair<pair<PointT, PointT>, PointT>> vector_nomal_x_rot;
    vector<pair<pair<PointT, PointT>, PointT>> vector_nomal_y_rot;
    vector<pair<pair<PointT, PointT>, PointT>> vector_nomal_z_rot_xy_trans;

    for (int r = 0; r < row_num_; ++r) {
      sort(surface_points_flat_temp_vec[r].begin(), surface_points_flat_temp_vec[r].end(), 
        [](const PointT &p_1, const PointT &p_2){
          return p_1.intensity <= p_2.intensity;
        });
    }

    for (int r = 0; r < row_num_; ++r) {
      sort(surface_points_flat_temp_vec_deskew[r].begin(), surface_points_flat_temp_vec_deskew[r].end(), 
        [](const PointT &p_1, const PointT &p_2){
          return p_1.intensity <= p_2.intensity;
        });
    }


// LOG(INFO) << "OK";
    for (int r = 0; r < row_num_; ++r) {

      // if (r  > 0) {
      //   kdtree_pre->setInputCloud(surface_points_flat_temp_vec_deskew[r - 1].makeShared());
      // }
      // if (r < row_num_ -1) {
      //   kdtree_next->setInputCloud(surface_points_flat_temp_vec_deskew[r + 1].makeShared());
      // }

      size_t ring_point_num = surface_points_flat_temp_vec[r].size();
      for (int k = 0; k < ring_point_num; ++k) {
        PointT point = surface_points_flat_temp_vec[r].points[k];
        PointT point_des = surface_points_flat_temp_vec_deskew[r].points[k];

        int ring_inf = int(point_des.intensity);
        float &time_inf = point_des.intensity;

        search_points.clear();
        dis_vector.clear();

        PointCloud &search_points_ring = surface_points_flat_temp_vec_deskew[r];

        search_points.push_back(point_des);
//LOG(INFO) << point_des;


        if (k == 0 && k < ring_point_num - 1) {
          search_points.push_back(search_points_ring[k + 1]);
        } else if (k == ring_point_num - 1 && k > 0) {
          search_points.push_back(search_points_ring[k - 1]);
        } else {
          // float dis_1 = CalcSquaredDiff(search_points_ring[k - 1], point_des);
          // float dis_2 = CalcSquaredDiff(search_points_ring[k + 1], point_des);
          // if (dis_1 < dis_2) {
          //   search_points.push_back(search_points_ring[k - 1]);
          // } else {
          //   search_points.push_back(search_points_ring[k + 1]);
          // }
          search_points.push_back(search_points_ring[k - 1]);
          search_points.push_back(search_points_ring[k + 1]);
        }

        if (r > 0) {
          PointCloud &search_points_previous_ring = surface_points_flat_temp_vec_deskew[r - 1];
          size_t ring_point_pre_num = search_points_previous_ring.size();
          
          if (ring_point_pre_num > 0) {
            auto iter = find_binary(search_points_previous_ring.begin(), search_points_previous_ring.end(), (time_inf-1));
            if (iter != search_points_previous_ring.end()) {
              search_points.push_back(*iter);
            }
            if (iter != search_points_previous_ring.begin()) {
              search_points.push_back(*(--iter));             
            }
          }
        }


        if (r < row_num_- 1) {
          PointCloud &search_points_next_ring = surface_points_flat_temp_vec_deskew[r + 1];
          size_t ring_point_next_num = search_points_next_ring.size();
          
          if (ring_point_next_num > 0) {
            auto iter = find_binary(search_points_next_ring.begin(), search_points_next_ring.end(), (time_inf+1));
            if (iter != search_points_next_ring.end()) {
              search_points.push_back(*iter);
            }
            if (iter != search_points_next_ring.begin()) {
              search_points.push_back(*(--iter));             
            }
          }
        }



        size_t search_points_num = search_points.size();
        if (search_points_num < 5) {
          continue;
        }

        for (int i = 0; i < search_points_num; ++i) {
          float dis = CalcSquaredDiff(point_des, search_points[i]);
          dis_pair.first = dis;
          dis_pair.second = search_points[i];
          dis_vector.push_back(dis_pair);
        }

        sort(dis_vector.begin(), dis_vector.end(), 
          [](const pair<float, PointT> &pair_1, const pair<float, PointT> &pair_2){
          return pair_1.first < pair_2.first;
        });



        Eigen::Matrix<float, 5, 3> matA0;
        Eigen::Matrix<float, 5, 1> matB0 = -1 * Eigen::Matrix<float, 5, 1>::Ones();

        PointT point_normal_1, point_normal_2, point_normal;


        if (dis_vector[4].first < config_.max_sq_dis) {



          std::vector<Eigen::Vector3d> nearCorners;
          Eigen::Vector3d center(0, 0, 0);
          for (int j = 0; j < 5; j++) {
            Eigen::Vector3d tmp(dis_vector[j].second.x,
                                dis_vector[j].second.y,
                                dis_vector[j].second.z);
            center += tmp;
            nearCorners.push_back(tmp);
          }
          center = center / 5.0;

          Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
          for (int j = 0; j < 5; j++)
          {
            Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
            covMat += (tmpZeroMean * tmpZeroMean.transpose());
          }

          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

          // if is indeed line feature
          // note Eigen library sort eigenvalues in increasing order
          Eigen::Vector3d unit_direction = saes.eigenvectors().col(0);

          if (saes.eigenvalues()[1] < 10 * saes.eigenvalues()[0]) {
            continue;             
          }
//LOG(INFO) << saes.eigenvectors().col(0);



          for (int j = 0; j < 5; j++)
          {
              matA0(j, 0) = dis_vector[j].second.x;
              matA0(j, 1) = dis_vector[j].second.y;
              matA0(j, 2) = dis_vector[j].second.z;
          }

          Eigen::Vector3f norm = matA0.colPivHouseholderQr().solve(matB0);
          float negative_OA_dot_norm = 1 / norm.norm();
          norm.normalize();


          bool planeValid = true;
          for (int j = 0; j < 5; j++)
          {
              if (fabs( norm(0) * dis_vector[j].second.x +
                        norm(1) * dis_vector[j].second.y +
                        norm(2) * dis_vector[j].second.z + negative_OA_dot_norm) > 0.2)
              {
                  planeValid = false;
                  break;
              }
          }

          if (planeValid == false) {
            continue;
          }

          //PointT point_normal;
          point_normal_1.x = norm(0);
          point_normal_1.y = norm(1);
          point_normal_1.z = norm(2);

          PointT tripod1 = dis_vector[0].second;
          PointT tripod2 = dis_vector[1].second;
          PointT tripod3 = dis_vector[2].second;
          PointT tripod4 = dis_vector[3].second;

      

          Eigen::Vector3f edge_1(tripod3.x - tripod1.x, tripod3.y - tripod1.y, tripod3.z - tripod1.z);
          Eigen::Vector3f edge_2(tripod4.x - tripod2.x, tripod4.y - tripod2.y, tripod4.z - tripod2.z);

          // Eigen::Vector3f edge_1(tripod3.x - tripod1.x, tripod3.y - tripod1.y, tripod3.z - tripod1.z);
          // Eigen::Vector3f edge_2(tripod3.x - tripod2.x, tripod3.y - tripod2.y, tripod3.z - tripod2.z);

          Eigen::Vector3f point_normal_2 = edge_1.cross(edge_2);
          point_normal_2.normalize();

          float norm_score = norm.dot(point_normal_2);

          if (norm_score < 0) {
            point_normal_2 = -point_normal_2;
          }

          // if (fabs(norm_score) < 0.9) {
          //   continue;
          // }

          Eigen::Vector3f point_normal_vec = norm + point_normal_2;
          point_normal_vec.normalize();

          point_normal.x = point_normal_2(0);
          point_normal.y = point_normal_2(1);
          point_normal.z = point_normal_2(2);
//LOG(INFO) << point_normal_2;
          pair_nomal.first.first =  point_normal;
          pair_nomal.first.second = point_des;
          pair_nomal.second = point;

          if (config_.lower_ring_num_z_trans <= (ring_inf + 1) && (ring_inf + 1) <= config_.upper_ring_num_z_trans) {
            vector_nomal_z_trans.push_back(pair_nomal);
          }
          if (config_.lower_ring_num_x_rot <= (ring_inf + 1) && (ring_inf + 1) <= config_.upper_ring_num_x_rot) {
            vector_nomal_x_rot.push_back(pair_nomal);
          }
          if (config_.lower_ring_num_y_rot <= (ring_inf + 1) && (ring_inf + 1) <= config_.upper_ring_num_y_rot) {
            vector_nomal_y_rot.push_back(pair_nomal);
          }
          if (config_.lower_ring_num_z_rot_xy_trans <= (ring_inf + 1) && (ring_inf + 1)<= config_.upper_ring_num_z_rot_xy_trans) {
            vector_nomal_z_rot_xy_trans.push_back(pair_nomal);
          }
        } else {
          continue;
        }
      }
    }

      size_t num_z_trans = vector_nomal_z_trans.size();
      size_t num_x_rot = vector_nomal_x_rot.size();
      size_t num_y_rot = vector_nomal_y_rot.size();
      size_t num_z_rot_xy_trans = vector_nomal_z_rot_xy_trans.size();

    //  LOG(INFO) << "num_z_rot_xy_trans: " << num_z_rot_xy_trans;
      sort(vector_nomal_z_rot_xy_trans.begin(), vector_nomal_z_rot_xy_trans.end(), 
        [](const pair<pair<PointT, PointT>, PointT> &pair_1, const pair<pair<PointT, PointT>, PointT> &pair_2){
          return fabs(pair_1.first.first.x) > fabs(pair_2.first.first.x);
        });

      float con_trs_x = 0;
      for (int i = 0, j = 0; i < num_z_rot_xy_trans && j < config_.flat_extract_num_x_trans; ++i, ++j) {

        surface_points_normal_.push_back(vector_nomal_z_rot_xy_trans[i].first.first);

        surface_points_flat_.push_back(vector_nomal_z_rot_xy_trans[i].second);
        surface_points_flat_z_rot_xy_trans_.push_back(vector_nomal_z_rot_xy_trans[i].second);
        con_trs_x += fabs(vector_nomal_z_rot_xy_trans[i].first.first.x);
      }


      sort(vector_nomal_z_rot_xy_trans.begin(), vector_nomal_z_rot_xy_trans.end(), 
        [](const pair<pair<PointT, PointT>, PointT> &pair_1, const pair<pair<PointT, PointT>, PointT> &pair_2){
          return fabs(pair_1.first.first.y) > fabs(pair_2.first.first.y);
        });

      float con_trs_y = 0;
      for (int i = 0, j = 0; i < num_z_rot_xy_trans && j < config_.flat_extract_num_y_trans; ++i, ++j) {

        surface_points_normal_.push_back(vector_nomal_z_rot_xy_trans[i].first.first);     

        surface_points_flat_.push_back(vector_nomal_z_rot_xy_trans[i].second);
        surface_points_flat_z_rot_xy_trans_.push_back(vector_nomal_z_rot_xy_trans[i].second);
        con_trs_y += fabs(vector_nomal_z_rot_xy_trans[i].first.first.y);
      }


      sort(vector_nomal_z_trans.begin(), vector_nomal_z_trans.end(), 
        [](const pair<pair<PointT, PointT>, PointT> &pair_1, const pair<pair<PointT, PointT>, PointT> &pair_2){
          return fabs(pair_1.first.first.z) > fabs(pair_2.first.first.z);
        });

      float con_trs_z = 0;
      for (int i = 0, j = 0; i < num_z_trans && j < config_.flat_extract_num_z_trans; ++i, ++j) {

        surface_points_normal_.push_back(vector_nomal_z_trans[i].first.first);

        surface_points_flat_.push_back(vector_nomal_z_trans[i].second);
        surface_points_flat_z_trans_.push_back(vector_nomal_z_trans[i].second);
        con_trs_z += fabs(vector_nomal_z_trans[i].first.first.z);
      }


      sort(vector_nomal_z_rot_xy_trans.begin(), vector_nomal_z_rot_xy_trans.end(),
        [](const pair<pair<PointT, PointT>, PointT>&pair_1, const pair<pair<PointT, PointT>, PointT> &pair_2){
          float corss_z_1 = pair_1.first.first.x * pair_1.first.second.y - pair_1.first.first.y * pair_1.first.second.x;
          float corss_z_2 = pair_2.first.first.x * pair_2.first.second.y - pair_2.first.first.y * pair_2.first.second.x;
        
          return fabs(corss_z_1) > fabs(corss_z_2);
        });

      float con_rot_z = 0;
      for (int i = 0, j = 0; i < num_z_rot_xy_trans && j < config_.flat_extract_num_z_rot; ++i, ++j) {
        surface_points_flat_.push_back(vector_nomal_z_rot_xy_trans[i].second);

        surface_points_normal_.push_back(vector_nomal_z_rot_xy_trans[i].first.first);

        surface_points_flat_z_rot_xy_trans_.push_back(vector_nomal_z_rot_xy_trans[i].second);
        con_rot_z += fabs(vector_nomal_z_rot_xy_trans[i].first.first.x * vector_nomal_z_rot_xy_trans[i].first.second.y - 
                          vector_nomal_z_rot_xy_trans[i].first.first.y * vector_nomal_z_rot_xy_trans[i].first.second.x);
      }


      sort(vector_nomal_x_rot.begin(), vector_nomal_x_rot.end(),
        [](const pair<pair<PointT, PointT>, PointT> &pair_1, const pair<pair<PointT, PointT>, PointT> &pair_2){
          float corss_x_1 = pair_1.first.first.y * pair_1.first.second.z - pair_1.first.first.z * pair_1.first.second.y;
          float corss_x_2 = pair_2.first.first.y * pair_2.first.second.z - pair_2.first.first.z * pair_2.first.second.y;
          return fabs(corss_x_1) > fabs(corss_x_2);
        });

      float con_rot_x = 0;
      for (int i = 0, j = 0; i < num_x_rot && j < config_.flat_extract_num_x_rot; ++i, ++j) {

        surface_points_normal_.push_back(vector_nomal_x_rot[i].first.first);

        surface_points_flat_.push_back(vector_nomal_x_rot[i].second);
        
        surface_points_flat_x_rot_.push_back(vector_nomal_x_rot[i].second);

        con_rot_x += fabs(vector_nomal_x_rot[i].first.first.y * vector_nomal_x_rot[i].first.second.z -
                          vector_nomal_x_rot[i].first.first.z * vector_nomal_x_rot[i].first.second.y);
      }


      sort(vector_nomal_y_rot.begin(), vector_nomal_y_rot.end(), 
        [](const pair<pair<PointT, PointT>, PointT> &pair_1, const pair<pair<PointT, PointT>, PointT> &pair_2){
          float corss_y_1 = pair_1.first.first.z * pair_1.first.second.x - pair_1.first.first.x * pair_1.first.second.z;
          float corss_y_2 = pair_2.first.first.z * pair_2.first.second.x - pair_2.first.first.x * pair_2.first.second.z;
          return fabs(corss_y_1) > fabs(corss_y_2);
        });

      float con_rot_y = 0;
      for (int i = 0, j = 0; i < num_y_rot && j < config_.flat_extract_num_y_rot; ++i, ++j) {

        surface_points_normal_.push_back(vector_nomal_y_rot[i].first.first);

        surface_points_flat_.push_back(vector_nomal_y_rot[i].second);
        
        surface_points_flat_y_rot_.push_back(vector_nomal_y_rot[i].second);

        con_rot_y += fabs(vector_nomal_y_rot[i].first.first.z * vector_nomal_y_rot[i].first.second.x - 
                          vector_nomal_y_rot[i].first.first.x * vector_nomal_y_rot[i].first.second.z);
      }
// LOG(INFO) << "OK";
  }




} // ExtractFeaturePoints






void PointProcessor::ExtractFeaturePoints_2() {

  std::pair<Eigen::Vector2i, Eigen::Vector3d> pair_point_index_flat_normal;
  std::vector<std::pair<Eigen::Vector2i, Eigen::Vector3d>> vector_pair_flat;

  std::pair<Eigen::Vector2i, float> pair_point_index_curve;
  std::vector<std::pair<Eigen::Vector2i, float>> vector_pair_curve;

  int skip_pixel_num = 3;

  for (int i = 0; i < row_num_; ++i) {

    vector_pair_curve.clear();

    for (int j = 0; j < col_num_; ++j) {

      if (range_image_[i][j].occupy_state == 1){
        if (CalcPointDistance(range_image_[i][j].point) < 3) {
          continue;
        }

        PointCloud flat_cloud;
        PointCloud row_curve_cloud;
        PointCloud col_curve_cloud;

        int row_start = i - 0;
        int row_end = i + 0;
        int col_start = j - 1;
        int col_end = j + 1;

        int row_curve_pixel_num = 4;
        int col_curve_pixel_num = 4;
        int row_flat_pixel_num = 2;
        int col_flat_pixel_num = 2;

        int current_right_pixel_num = 0;
        int current_left_pixel_num = 0;
        int current_down_pixel_num = 0;
        int current_up_pixel_num = 0;
        int pixel_num = 1;

        if (row_start < 0)
          row_start = 0;
        if (row_end > row_num_ - 1)
          row_end = row_num_ -1;

        if (col_start < 0)
          col_start = 0;
        if (col_end > col_num_ - 1)
          col_end = col_num_ - 1;
//LOG(INFO) << i << " " << j;
        flat_cloud.push_back(range_image_[i][j].point);

        for (int m = row_start; m < row_end + 1; ++m) {
          for (int n = col_start; n < col_end + 1; ++n) {
            if (range_image_[m][n].occupy_state == 1) {
              if (m != i || n != j) {
                if (m < i && current_down_pixel_num < col_curve_pixel_num) {
                  ++current_down_pixel_num;
                  col_curve_cloud.push_back(range_image_[m][n].point);
                }
                if (m > i && current_up_pixel_num < col_curve_pixel_num) {
                  ++current_up_pixel_num;
                  col_curve_cloud.push_back(range_image_[m][n].point);
                }
                if (n < j && current_right_pixel_num < row_curve_pixel_num) {
                  ++current_right_pixel_num;
                  row_curve_cloud.push_back(range_image_[m][n].point);
                }
                if (n > j && current_left_pixel_num < row_curve_pixel_num) {
                  ++current_left_pixel_num;
                  row_curve_cloud.push_back(range_image_[m][n].point);
                }
                ++pixel_num;
                flat_cloud.push_back(range_image_[m][n].point);  
              }
            }
          }
        }

        if (current_right_pixel_num < row_curve_pixel_num) {
          for (int n = col_start - 1; n + 1 > 0; --n) {
            for (int m = row_start; m < row_end + 1; ++m) {
              if (range_image_[m][n].occupy_state == 1) {
                if (current_right_pixel_num < row_flat_pixel_num) {
                  ++pixel_num;
                  flat_cloud.push_back(range_image_[m][n].point);     
                }
                ++current_right_pixel_num;
                row_curve_cloud.push_back(range_image_[m][n].point);
//LOG(INFO) << m << " " << n;
              }
              if (current_right_pixel_num == row_curve_pixel_num)
                break;
            }
            if (current_right_pixel_num == row_curve_pixel_num)
              break;
          }
        }
//LOG(INFO) << current_right_pixel_num;

        if (current_left_pixel_num < row_curve_pixel_num) {
          for (int n = col_end + 1; n < col_num_; ++n) {
            for (int m = row_start; m < row_end + 1; ++m) {
              if (range_image_[m][n].occupy_state == 1) {
                if (current_left_pixel_num < row_flat_pixel_num) {
                  ++pixel_num;
                  flat_cloud.push_back(range_image_[m][n].point);     
                }
                ++current_left_pixel_num;
                row_curve_cloud.push_back(range_image_[m][n].point);
//LOG(INFO) << m << " " << n;
              }
              if (current_left_pixel_num == row_curve_pixel_num)
                break;
            }
            if (current_left_pixel_num == row_curve_pixel_num)
              break;
          }
        }
//LOG(INFO) << current_left_pixel_num;

        if (current_down_pixel_num < col_curve_pixel_num) {
          for (int m = row_start - 1; m + 1 > 0; --m) {
            for (int n = col_start; n < col_end + 1; ++n) {
              if (range_image_[m][n].occupy_state == 1) {
                if (current_down_pixel_num < col_flat_pixel_num) {
                  ++pixel_num;
                  flat_cloud.push_back(range_image_[m][n].point);     
                }                
                ++current_down_pixel_num;
                col_curve_cloud.push_back(range_image_[m][n].point);
//LOG(INFO) << m << " " << n;
              }
              if (current_down_pixel_num == col_curve_pixel_num)
                break;
            }
            if (current_down_pixel_num == col_curve_pixel_num)
              break;
          }
        }
//LOG(INFO) << current_down_pixel_num;

        if (current_up_pixel_num < col_curve_pixel_num) {
          for (int m = row_end + 1; m < row_num_; ++m) {
            for (int n = col_start; n < col_end + 1; ++n) {
              if (range_image_[m][n].occupy_state == 1) {
                if (current_up_pixel_num < col_flat_pixel_num) {
                  ++pixel_num;
                  flat_cloud.push_back(range_image_[m][n].point);     
                }   
                ++current_up_pixel_num;
                col_curve_cloud.push_back(range_image_[m][n].point);
//LOG(INFO) << m << " " << n;
              }
              if (current_up_pixel_num == col_curve_pixel_num)
                break;
            }
            if (current_up_pixel_num == col_curve_pixel_num)
              break;
          }
        }
//LOG(INFO) << current_up_pixel_num;



        float curve;
        if (current_right_pixel_num == current_left_pixel_num &&
            current_right_pixel_num == row_curve_pixel_num) {
          float point_distance  = CalcPointDistance(range_image_[i][j].point);
          point_distance = -2 * row_curve_pixel_num * point_distance;            
//LOG(INFO) << current_right_pixel_num << " " << current_down_pixel_num;
          for (int n = 0; n < 2 * row_curve_pixel_num; ++n) {
            point_distance  += CalcPointDistance(row_curve_cloud[n]);
          }
          curve = pow(point_distance, 2);
//LOG(INFO) << curve;
        } else {
          continue;
        }


//LOG(INFO) << curve;

        if (curve < 0.1) {

//LOG(INFO) << flat_cloud.size() << " " << pixel_num;
          Eigen::Matrix<double, Eigen::Dynamic, 3> matA0(pixel_num, 3);
          Eigen::Matrix<double, Eigen::Dynamic, 1> matB0(pixel_num, 1);


          for(int r = 0; r < pixel_num; ++r) {
            matA0(r, 0) = double(flat_cloud[r].x);
            matA0(r, 1) = double(flat_cloud[r].y);
            matA0(r, 2) = double(flat_cloud[r].z);
            matB0(r, 0) = -1.0;
          }

// LOG(INFO) << matA0;
          if (pixel_num > 2) {
            Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
            double negative_OA_dot_norm = 1 / norm.norm();
            norm.normalize();          

            bool plane_valid = true;
            for (int r = 0; r < pixel_num; ++r) {
              if (fabs( norm(0) * matA0(r, 0) +
                        norm(1) * matA0(r, 1) +
                        norm(2) * matA0(r, 2) + 
                        negative_OA_dot_norm) > 0.2) {

                plane_valid = false;
                break;
              }
            }

            if (plane_valid) {
              pair_point_index_flat_normal.first(0) = i;
              pair_point_index_flat_normal.first(1) = j;
              pair_point_index_flat_normal.second = norm;
              vector_pair_flat.push_back(pair_point_index_flat_normal);
            }
          }
        } else {
          pair_point_index_curve.first(0) = i;
          pair_point_index_curve.first(1) = j;
          pair_point_index_curve.second = curve;
          vector_pair_curve.push_back(pair_point_index_curve);
        }
      }
    }


    sort (vector_pair_curve.begin(), vector_pair_curve.end(),
      [](const std::pair<Eigen::Vector2i, float>& p_1, const std::pair<Eigen::Vector2i, float>& p_2){
        return p_1.second > p_2.second;
      });



    for (int num_1 = 0; num_1 < vector_pair_curve.size() && num_1 < config_.max_corner_less_sharp; ++num_1) {
      if (range_image_[vector_pair_curve[num_1].first(0)][vector_pair_curve[num_1].first(1)].feature_state == 0) {
        if (num_1 < config_.max_corner_sharp) {
          corner_points_sharp_.push_back(range_image_[vector_pair_curve[num_1].first(0)][vector_pair_curve[num_1].first(1)].point);
        }

        range_image_[vector_pair_curve[num_1].first(0)][vector_pair_curve[num_1].first(1)].feature_state = 1;
        for (int num_2 = 1; num_2 < config_.num_feature_regions; ++num_2) {
          if(vector_pair_curve[num_1].first(1) + num_2 < col_num_) {
            range_image_[vector_pair_curve[num_1].first(0)][vector_pair_curve[num_1].first(1) + num_2].feature_state = 1;
          }
          if(vector_pair_curve[num_1].first(1) - num_2 + 1 > 0) {
            range_image_[vector_pair_curve[num_1].first(0)][vector_pair_curve[num_1].first(1) - num_2].feature_state = 1;
          }
        }
        corner_points_less_sharp_.push_back(range_image_[vector_pair_curve[num_1].first(0)][vector_pair_curve[num_1].first(1)].point);
      }
    }



  }








//   sort (vector_pair_curve.begin(), vector_pair_curve.end(),
//     [](const std::pair<Eigen::Vector2i, float>& p_1, const std::pair<Eigen::Vector2i, float>& p_2){
//       return p_1.second > p_2.second;
//     });



// for (int i = 0; i < vector_pair_curve.size(); i++) {
//   corner_points_sharp_.push_back(range_image_[vector_pair_curve[i].first(0)][vector_pair_curve[i].first(1)]);
// LOG(INFO) << vector_pair_curve[i].second;
// }


  std::vector<std::pair<Eigen::Vector2i, Eigen::Vector3d>> vector_less_flat;
  for (int num_1 = 0; num_1 < vector_pair_flat.size(); ++num_1) {
    if (range_image_[vector_pair_flat[num_1].first(0)][vector_pair_flat[num_1].first(1)].feature_state == 0) {
      vector_less_flat.push_back(vector_pair_flat[num_1]);
      range_image_[vector_pair_flat[num_1].first(0)][vector_pair_flat[num_1].first(1)].feature_state = 1;

      for (int num_2 = 1; num_2 < config_.num_feature_regions; ++num_2) {
        if(vector_pair_flat[num_1].first(1) + num_2 < col_num_) {
          range_image_[vector_pair_flat[num_1].first(0)][vector_pair_flat[num_1].first(1) + num_2].feature_state = 1;
        }
        if(vector_pair_flat[num_1].first(1) - num_2 + 1 > 0) {
          range_image_[vector_pair_flat[num_1].first(0)][vector_pair_flat[num_1].first(1) - num_2].feature_state = 1;
        }
      }
    }
  }


  for (int num = 0; num < vector_less_flat.size(); ++num) {
    //range_image_[vector_less_flat[num].first(0)][vector_less_flat[num].first(1)].intensity = 0;
    surface_points_less_flat_.push_back(range_image_[vector_less_flat[num].first(0)][vector_less_flat[num].first(1)].point);
  }


  sort (vector_less_flat.begin(), vector_less_flat.end(),
    [] (const std::pair<Eigen::Vector2i, Eigen::Vector3d> &pair_1, const std::pair<Eigen::Vector2i, Eigen::Vector3d> &pair_2) {
      return fabs(pair_1.second(0)) > fabs(pair_2.second(0));
    });

  for (int num = 0, j = 0; num < vector_less_flat.size() && j < config_.flat_extract_num_x_trans; ++num) {
    int row = vector_less_flat[num].first(0) + 1;
    if ( config_.lower_ring_num_z_rot_xy_trans <= row && row <= config_.upper_ring_num_z_rot_xy_trans) {
        ++j;
        PointT norm;
        norm.x = vector_less_flat[num].second(0);
        norm.y = vector_less_flat[num].second(1);
        norm.z = vector_less_flat[num].second(2);

        surface_points_normal_.push_back(norm);
        surface_points_flat_.push_back(range_image_[vector_less_flat[num].first(0)][vector_less_flat[num].first(1)].point);
    }
  }


  sort (vector_less_flat.begin(), vector_less_flat.end(),
    [] (const std::pair<Eigen::Vector2i, Eigen::Vector3d> &pair_1, const std::pair<Eigen::Vector2i, Eigen::Vector3d> &pair_2) {
      return fabs(pair_1.second(1)) > fabs(pair_2.second(1));
    });

  for (int num = 0, j = 0; num < vector_less_flat.size() && j < config_.flat_extract_num_y_trans; ++num) {
    int row = vector_less_flat[num].first(0) + 1;
    if ( config_.lower_ring_num_z_rot_xy_trans <= row && row <= config_.upper_ring_num_z_rot_xy_trans) {
        ++j;
        PointT norm;
        norm.x = vector_less_flat[num].second(0);
        norm.y = vector_less_flat[num].second(1);
        norm.z = vector_less_flat[num].second(2);

        surface_points_normal_.push_back(norm);
        surface_points_flat_.push_back(range_image_[vector_less_flat[num].first(0)][vector_less_flat[num].first(1)].point);
    }
  }


  sort (vector_less_flat.begin(), vector_less_flat.end(),
    [] (const std::pair<Eigen::Vector2i, Eigen::Vector3d> &pair_1, const std::pair<Eigen::Vector2i, Eigen::Vector3d> &pair_2) {
      return fabs(pair_1.second(2)) > fabs(pair_2.second(2));
    });

  for (int num = 0, j = 0; num < vector_less_flat.size() && j < config_.flat_extract_num_z_trans; ++num) {
    int row = vector_less_flat[num].first(0) + 1;
    if ( config_.lower_ring_num_z_trans <= row && row <= config_.upper_ring_num_z_trans) {
        ++j;
        PointT norm;
        norm.x = vector_less_flat[num].second(0);
        norm.y = vector_less_flat[num].second(1);
        norm.z = vector_less_flat[num].second(2);

        surface_points_normal_.push_back(norm);
        surface_points_flat_.push_back(range_image_[vector_less_flat[num].first(0)][vector_less_flat[num].first(1)].point);
    }
  }


  vector<vector<ImageElement>> &range_image = range_image_;
  sort (vector_less_flat.begin(), vector_less_flat.end(),
    [&range_image] (const std::pair<Eigen::Vector2i, Eigen::Vector3d> &pair_1, const std::pair<Eigen::Vector2i, Eigen::Vector3d> &pair_2) {
      PointT &point_1 = range_image[pair_1.first(0)][pair_1.first(1)].point;
      PointT &point_2 = range_image[pair_2.first(0)][pair_2.first(1)].point;

      float corss_x_1 = pair_1.second(1) * point_1.z - pair_1.second(2) *  point_1.y;
      float corss_x_2 = pair_2.second(1) * point_2.z - pair_2.second(2) *  point_2.y;
      return fabs(corss_x_1) > fabs(corss_x_2);
    });

  for (int num = 0, j = 0; num < vector_less_flat.size() && j < config_.flat_extract_num_x_rot; ++num) {
    int row = vector_less_flat[num].first(0) + 1;
    if ( config_.lower_ring_num_x_rot <= row && row <= config_.upper_ring_num_x_rot) {
        ++j;
        PointT norm;
        norm.x = vector_less_flat[num].second(0);
        norm.y = vector_less_flat[num].second(1);
        norm.z = vector_less_flat[num].second(2);

        surface_points_normal_.push_back(norm);
        surface_points_flat_.push_back(range_image_[vector_less_flat[num].first(0)][vector_less_flat[num].first(1)].point);
    }
  }


  sort (vector_less_flat.begin(), vector_less_flat.end(),
    [&range_image] (const std::pair<Eigen::Vector2i, Eigen::Vector3d> &pair_1, const std::pair<Eigen::Vector2i, Eigen::Vector3d> &pair_2) {
      PointT &point_1 = range_image[pair_1.first(0)][pair_1.first(1)].point;
      PointT &point_2 = range_image[pair_2.first(0)][pair_2.first(1)].point;

      float corss_y_1 = pair_1.second(2) * point_1.x - pair_1.second(0) *  point_1.z;
      float corss_y_2 = pair_2.second(2) * point_2.x - pair_2.second(0) *  point_2.z;
      return fabs(corss_y_1) > fabs(corss_y_2);
    });

  for (int num = 0, j = 0; num < vector_less_flat.size() && j < config_.flat_extract_num_y_rot; ++num) {
    int row = vector_less_flat[num].first(0) + 1;
    if ( config_.lower_ring_num_y_rot <= row && row <= config_.upper_ring_num_y_rot) {
        ++j;
        PointT norm;
        norm.x = vector_less_flat[num].second(0);
        norm.y = vector_less_flat[num].second(1);
        norm.z = vector_less_flat[num].second(2);

        surface_points_normal_.push_back(norm);
        surface_points_flat_.push_back(range_image_[vector_less_flat[num].first(0)][vector_less_flat[num].first(1)].point);
    }
  }


  sort (vector_less_flat.begin(), vector_less_flat.end(),
    [&range_image] (const std::pair<Eigen::Vector2i, Eigen::Vector3d> &pair_1, const std::pair<Eigen::Vector2i, Eigen::Vector3d> &pair_2) {
      PointT &point_1 = range_image[pair_1.first(0)][pair_1.first(1)].point;
      PointT &point_2 = range_image[pair_2.first(0)][pair_2.first(1)].point;

      float corss_z_1 = pair_1.second(0) * point_1.y - pair_1.second(1) *  point_1.x;
      float corss_z_2 = pair_2.second(0) * point_2.y - pair_2.second(1) *  point_2.x;
      return fabs(corss_z_1) > fabs(corss_z_2);
    });

  for (int num = 0, j = 0; num < vector_less_flat.size() && j < config_.flat_extract_num_z_rot; ++num) {
    int row = vector_less_flat[num].first(0) + 1;
    if ( config_.lower_ring_num_z_rot_xy_trans <= row && row <= config_.lower_ring_num_z_rot_xy_trans) {
        ++j;
        PointT norm;
        norm.x = vector_less_flat[num].second(0);
        norm.y = vector_less_flat[num].second(1);
        norm.z = vector_less_flat[num].second(2);

        surface_points_normal_.push_back(norm);
        surface_points_flat_.push_back(range_image_[vector_less_flat[num].first(0)][vector_less_flat[num].first(1)].point);
    }
  }


//   for (int i = 0; i < row_num_; ++i) {
//     for (int j = 0; j < col_num_; ++j) {
//       if (range_image_[i][j].occupy_state == 1){
//         cloud_in_rings_.push_back(range_image_[i][j].point);
//       }

//     }
//   }


// LOG(INFO) << cloud_in_rings_.size();
} // ExtractFeaturePoints_2








void PointProcessor::PublishResults() {

  if (!is_ros_setup_) {
    DLOG(WARNING) << "ros is not set up, and no results will be published";
    return;
  }

  // publish full resolution and feature point clouds
  PublishCloudMsg(pub_full_cloud_, cloud_in_rings_, sweep_start_, config_.capture_frame_id);

  PublishCloudMsg(pub_corner_points_sharp_, corner_points_sharp_, sweep_start_, config_.capture_frame_id);
  PublishCloudMsg(pub_corner_points_less_sharp_, corner_points_less_sharp_, sweep_start_, config_.capture_frame_id);

  PublishCloudMsg(pub_surf_points_flat_, surface_points_flat_, sweep_start_, config_.capture_frame_id);
  PublishCloudMsg(pub_surf_points_less_flat_, surface_points_less_flat_, sweep_start_, config_.capture_frame_id);

  PublishCloudMsg(pub_surf_points_flat_z_trans_, surface_points_flat_z_trans_, sweep_start_, config_.capture_frame_id);
  PublishCloudMsg(pub_surf_points_flat_z_rot_xy_trans_, surface_points_flat_z_rot_xy_trans_, sweep_start_, config_.capture_frame_id);
  PublishCloudMsg(pub_surf_points_flat_x_rot_, surface_points_flat_x_rot_, sweep_start_, config_.capture_frame_id);
  PublishCloudMsg(pub_surf_points_flat_y_rot_, surface_points_flat_y_rot_, sweep_start_, config_.capture_frame_id);

  PublishCloudMsg(pub_surf_points_normal_, surface_points_normal_, sweep_start_, config_.capture_frame_id);


}

} // namespace lom
