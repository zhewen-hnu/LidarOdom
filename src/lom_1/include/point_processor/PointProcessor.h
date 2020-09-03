#ifndef LOM_POINTPROCESSOR_H_
#define LOM_POINTPROCESSOR_H_

#include <glog/logging.h>

#include <pcl/filters/voxel_grid.h>

#include <pcl/filters/filter.h>

#include <pcl/features/normal_3d.h>

#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/pcl_macros.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud.h>

#include "../../include/utils/common_ros.h"
#include "../../include/utils/TicToc.h"
#include "../../include/utils/math_utils.h"
#include "../../include/utils/geometry_utils.h"

#include "../../include/utils/CircularBuffer.h"

#include "point_types.h"

#include <std_msgs/Float32.h>

namespace lom {

using namespace std;
using namespace mathutils;
typedef pcl::PointXYZI PointT;
typedef typename pcl::PointCloud<PointT> PointCloud;
typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef typename pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;

typedef lom::PointXYZIR PointIR;

typedef std::pair<size_t, size_t> IndexRange;

// adapted from LOAM
/** Point label options. */
enum PointLabel {
  CORNER_SHARP = 2,       ///< sharp corner point
  CORNER_LESS_SHARP = 1,  ///< less sharp corner point
  SURFACE_LESS_FLAT = 0,  ///< less flat surface point
  SURFACE_FLAT = -1       ///< flat surface point
};

struct ImageElement {
  PointT point;
  int occupy_state = 0;
  int feature_state = 0;
};

struct PointProcessorConfig {
  bool deskew = false;
  bool even = true;
  bool using_livox = false;
  int sensor_type = 16;
  double deg_diff = 0.2;
  double scan_period = 0.1;
  double lower_bound = -15.0;
  double upper_bound = 15.0;
  int using_lower_ring_num = 1;
  int using_upper_ring_num = 16;
  int lower_ring_num_sharp_point = 6;
  int upper_ring_num_sharp_point = 16;
  int lower_ring_num_z_trans = 1;
  int upper_ring_num_z_trans = 6;
  int lower_ring_num_x_rot = 1;
  int upper_ring_num_x_rot = 6;
  int lower_ring_num_y_rot = 1;
  int upper_ring_num_y_rot = 6;  
  int lower_ring_num_z_rot_xy_trans = 6;
  int upper_ring_num_z_rot_xy_trans = 16;
  int num_scan_subregions = 8;
  int num_curvature_regions_corner = 1;
  int num_curvature_regions_flat = 5;
  int num_feature_regions = 5;
  double surf_curv_th = 0.01;
  double sharp_curv_th = 0.1;
  int max_corner_sharp = 2;
  int max_corner_less_sharp = 20;
  int max_surf_flat = 4;
  int max_surf_nomal = 30;
  double max_sq_dis = 25.0;
  int flat_extract_num = 100;
  int flat_extract_num_x_trans = 100;
  int flat_extract_num_y_trans = 100;
  int flat_extract_num_z_trans = 100;
  int flat_extract_num_x_rot = 100;
  int flat_extract_num_y_rot = 100;
  int flat_extract_num_z_rot = 100;
  double less_flat_filter_size = 0.2;
  bool infer_start_ori = false;
  string capture_frame_id = "/map";
  bool using_surf_point_normal = true;
  bool using_surf_point_2 = true;
  bool extract_per_ring = true;

  void setup_param(ros::NodeHandle &nh);
};

class PointProcessor {

 public:

  PointProcessor() = delete;
  PointProcessor(const PointProcessorConfig &config);
  // WARNING: is it useful to separate Process and SetInputCloud?
  // void Process(const PointCloudConstPtr &cloud_in, PointCloud &cloud_out);
  void Process();

  void PointCloudHandler(const sensor_msgs::PointCloud2ConstPtr &raw_points_msg);
  void PointCloudHandler(const sensor_msgs::PointCloudConstPtr &raw_points_msg);

  void TansfHandler(const nav_msgs::OdometryConstPtr &transf_se_msg);

  void SetupRos(ros::NodeHandle &nh);

  void SetInputCloud(const PointCloudConstPtr &all_cloud_in, ros::Time time_in = ros::Time::now());
  void SetInputCloud(const pcl::PointCloud<PointIR>::Ptr &all_cloud_in, ros::Time time_in = ros::Time::now());

  void PointToRing();
  void PointToRing(const PointCloudConstPtr &all_cloud_in,
                   vector<PointCloudPtr> &all_rings,
                   vector<PointCloudPtr> &all_intensity);

  void PointToRing(const pcl::PointCloud<PointIR>::Ptr &all_cloud_in,
                   vector<PointCloudPtr> &all_rings,
                   vector<PointCloudPtr> &all_intensity);

  void PointToRing(const PointCloudConstPtr &all_cloud_in,
                   vector<PointCloudPtr> &all_rings);

  inline int ElevationToRing(float rad) {
    // if (config_.sensor_type == 16) {
    //   return int((RadToDeg(rad) - config_.lower_bound) * factor_ + 0.5);
    // }
    // else if (config_.sensor_type == 32) {
    //   return int((RadToDeg(rad) + 92.0/3.0) * 3.0 / 4.0);
    // }
    // else if (config_.sensor_type == 64) {   
    //   if (RadToDeg(rad) > 2.0 - 31 / 3.0)
    //     return (63 - int((2 - RadToDeg(rad)) * 3.0 + 0.5));
    //   else {
    //     return (32 - int((2.0 - 31 / 3.0 - RadToDeg(rad)) * 2.0 + 0.5));
    //   }
    // }

    return int((RadToDeg(rad) - config_.lower_bound) * factor_ + 0.5);

  }

  // TODO: Add IMU compensation, by quaternion interpolation
  // TODO: To implement, for better feature extraction
  // WARNING: Adding IMU here will require adding IMU in the odometry part
  void DeSkew(PointCloud &points);

  void ExtractFeaturePoints();

  void ExtractFeaturePoints_2();

  void PublishResults();

  // TODO: not necessary data?
  vector<PointCloudPtr> all_laser_scans_;
  vector<PointCloudPtr> all_intensity_scans_;

  vector<PointCloudPtr> laser_scans_;
  vector<PointCloudPtr> intensity_scans_;
  vector<IndexRange> scan_ranges_;
  vector<vector<ImageElement>> range_image_;


 protected:

  ros::Time sweep_start_;
  ros::Time scan_time_;

  int row_num_;
  int col_num_;

  double factor_;
  float time_factor_;

  PointProcessorConfig config_;
  TicToc tic_toc_;

  PointCloudConstPtr all_cloud_ptr_;
  pcl::PointCloud<PointIR>::Ptr all_cloud_ir_ptr_;

  PointCloud cloud_in_rings_;

  PointCloud corner_points_sharp_;
  PointCloud corner_points_less_sharp_;
  PointCloud surface_points_flat_;
  PointCloud surface_points_less_flat_;
  PointCloud surface_points_flat_z_trans_;
  PointCloud surface_points_flat_z_rot_xy_trans_;
  PointCloud surface_points_flat_x_rot_;
  PointCloud surface_points_flat_y_rot_;
  PointCloud surface_points_normal_;

  Eigen::Vector3f transf_se_pos_;
  Eigen::Quaternionf transf_se_rot_;
  ros::Time transf_se_time_;

  // the following will be assigened or resized
  vector<int> scan_ring_mask_;
  vector<pair<float, size_t> > curvature_idx_pairs_; // in subregion
  vector<PointLabel> subregion_labels_;     ///< point label buffer

//  void PrepareRing(const size_t idx_ring);
//  void PrepareSubregion(const size_t idx_ring, const size_t idx_start, const size_t idx_end);
//  void MaskPickedInRing(const size_t idx_ring, const size_t in_scan_idx);

  void Reset(const ros::Time &scan_time, const bool &is_new_sweep = true);
  void PrepareRing_corner(const PointCloud &scan);
  void PrepareRing_corner(const vector<PointCloudPtr> &scan);
  void PrepareRing_flat(const PointCloud &scan);
  void PrepareSubregion_corner(const PointCloud &scan, const size_t idx_start, const size_t idx_end);
  void PrepareSubregion_flat(const PointCloud &scan, const size_t idx_start, const size_t idx_end);
  void MaskPickedInRing(const PointCloud &scan, const size_t in_scan_idx);

  PointCloud::iterator find_binary(const PointCloud::iterator &begin, const PointCloud::iterator &end, const float &intensity);
  ros::Subscriber sub_raw_points_;   ///< input cloud message subscriber
  ros::Subscriber sub_tansf_es_;

  ros::Publisher pub_full_cloud_;              ///< full resolution cloud message publisher
  ros::Publisher pub_corner_points_sharp_;       ///< sharp corner cloud message publisher
  ros::Publisher pub_corner_points_less_sharp_;   ///< less sharp corner cloud message publisher
  ros::Publisher pub_surf_points_flat_;           ///< flat surface cloud message publisher
  ros::Publisher pub_surf_points_less_flat_;      ///< less flat surface cloud message publisher
  ros::Publisher pub_surf_points_flat_z_trans_;
  ros::Publisher pub_surf_points_flat_z_rot_xy_trans_;
  ros::Publisher pub_surf_points_flat_x_rot_;
  ros::Publisher pub_surf_points_flat_y_rot_;
  ros::Publisher pub_surf_points_normal_;

  bool is_ros_setup_ = false;

 private:
  float start_ori_, end_ori_;
  CircularBuffer<float> start_ori_buf1_{10};
  CircularBuffer<float> start_ori_buf2_{10};
  ros::Publisher pub_start_ori_;
  ros::Publisher pub_start_ori_inferred_;

};

} // namespace lom

#endif //LOM_POINTPROCESSOR_H_
