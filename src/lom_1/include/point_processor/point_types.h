#ifndef LOM_POINT_TYPES_H_
#define LOM_POINT_TYPES_H_

#include <pcl/point_types.h>

namespace lom {

/** Euclidean coordinate, including intensity and ring number. */
struct PointXYZIR
{
  PCL_ADD_POINT4D;                    // quad-word XYZ
  float    intensity;                 ///< laser intensity reading
  uint16_t ring;                      ///< laser ring number
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // ensure proper alignment
}
EIGEN_ALIGN16;
}  // namespace lom

POINT_CLOUD_REGISTER_POINT_STRUCT(lom::PointXYZIR,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (uint16_t, ring, ring))

#endif  // LOM_POINT_TYPES_H_