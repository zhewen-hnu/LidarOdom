#ifndef LOM_LOADVIRTUAL_H_
#define LOM_LOADVIRTUAL_H_

#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include "utils/CircularBuffer.h"

namespace lom_test {

struct MotionData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double timestamp;
  Eigen::Matrix3d Rwb;
  Eigen::Vector3d twb;
  Eigen::Vector3d imu_acc;
  Eigen::Vector3d imu_gyro;

  Eigen::Vector3d imu_gyro_bias;
  Eigen::Vector3d imu_acc_bias;

  Eigen::Vector3d imu_velocity;
};

void LoadPoseVel(std::string filename, std::vector<MotionData> &pose) {

  std::ifstream f;
  f.open(filename.c_str());

  if (!f.is_open()) {
    std::cerr << " can't open LoadFeatures file " << std::endl;
    return;
  }

  while (!f.eof()) {

    std::string s;
    std::getline(f, s);

    if (!s.empty()) {
      std::stringstream ss;
      ss << s;

      MotionData data;
      double time;
      Eigen::Quaterniond q;
      Eigen::Vector3d t;
      Eigen::Vector3d gyro;
      Eigen::Vector3d acc;

      Eigen::Vector3d vel;
      Eigen::Vector3d acc_bias;
      Eigen::Vector3d gyro_bias;

      ss >> time;
      ss >> q.w();
      ss >> q.x();
      ss >> q.y();
      ss >> q.z();
      ss >> t(0);
      ss >> t(1);
      ss >> t(2);
      ss >> vel(0);
      ss >> vel(1);
      ss >> vel(2);
      ss >> gyro(0);
      ss >> gyro(1);
      ss >> gyro(2);
      ss >> acc(0);
      ss >> acc(1);
      ss >> acc(2);
      ss >> acc_bias(0);
      ss >> acc_bias(1);
      ss >> acc_bias(2);
      ss >> gyro_bias(0);
      ss >> gyro_bias(1);
      ss >> gyro_bias(2);

      data.timestamp = time;
      data.imu_gyro = gyro;
      data.imu_acc = acc;
      data.twb = t;
      data.Rwb = Eigen::Matrix3d(q);

      data.imu_velocity = vel;
      data.imu_acc_bias = acc_bias;
      data.imu_gyro_bias = gyro_bias;

      pose.push_back(data);

    }
  }
}

}

#endif //LOM_LOADVIRTUAL_H_
