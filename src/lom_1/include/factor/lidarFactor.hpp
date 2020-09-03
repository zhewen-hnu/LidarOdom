#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include "utils/math_utils.h"

class LidarEdgeFactor_z_rot_xy_trans : public ceres::SizedCostFunction<1, 4, 3>
{
    public:

	LidarEdgeFactor_z_rot_xy_trans(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s) {}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {
		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lpa{last_point_a.x(), last_point_a.y(), last_point_a.z()};
		Eigen::Matrix<double, 3, 1> lpb{last_point_b.x(), last_point_b.y(), last_point_b.z()};
        Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};

		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

		Eigen::Matrix<double, 3, 1> lp = q_last_curr * cp + t_last_curr;
		Eigen::Matrix<double, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<double, 3, 1> nor = nu.cross(lpa - lpb);
		nor.normalize();

		residuals[0] = nor.dot(lp - lpa);

		if (jacobians) 
		{
			if (jacobians[0])
			{
				// jacobians[0][0] = 0;
				// jacobians[0][1] = 0;
				// jacobians[0][2] = -nor(0) * (q_last_curr * cp)(1) + nor(1) * (q_last_curr * cp)(0);
				// jacobians[0][3] = 0;




				Eigen::Matrix3d ror_mat = q_last_curr.toRotationMatrix();
                Eigen::Matrix3d point_mat = mathutils::SkewSymmetric(cp);
				
				Eigen::Matrix3d right_dev = -ror_mat * point_mat;

				Eigen::Matrix<double, 1, 3> result = nor.transpose() * right_dev;


				jacobians[0][0] = 0;
				jacobians[0][1] = 0;
				jacobians[0][2] = result(0, 2);
				jacobians[0][3] = 0;


			}
			if (jacobians[1]) 
			{
				jacobians[1][0] = nor(0);
				jacobians[1][1] = nor(1);
				jacobians[1][2] = 0;
			}
		}

		return true;
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};





struct LidarEdgeFactor
{
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp = q_last_curr * cp + t_last_curr;



		// Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		// Eigen::Matrix<T, 3, 1> de = lpa - lpb;
		// residual[0] = nu.x() / de.norm();
		// residual[1] = nu.y() / de.norm();
		// residual[2] = nu.z() / de.norm();




		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> nor = nu.cross(lpa - lpb);
		nor.normalize();
		residual[0] = nor.dot(lp - lpa);


		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactor, 1, 4, 3>(
			new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};




struct LidarEdgeFactorLast
{
	LidarEdgeFactorLast(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		//Eigen::Matrix<T, 3, 1> lp = q_last_curr * cp + t_last_curr;
        Eigen::Matrix<T, 3, 1> lp = q_last_curr.inverse() * (cp - t_last_curr);

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> nor = nu.cross(lpa - lpb);
		nor.normalize();
		residual[0] = nor.dot(lp - lpa);


		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactorLast, 1, 4, 3>(
			new LidarEdgeFactorLast(curr_point_, last_point_a_, last_point_b_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};




class LidarPlaneFactor_z_rot_xy_trans : public ceres::SizedCostFunction<1, 4, 3>
{
    public:

	LidarPlaneFactor_z_rot_xy_trans(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {


		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lpj{last_point_j.x(), last_point_j.y(), last_point_j.z()};
		Eigen::Matrix<double, 3, 1> ljm{ljm_norm.x(), ljm_norm.y(), ljm_norm.z()};

        Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};

		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

		Eigen::Matrix<double, 3, 1> lp = q_last_curr * cp + t_last_curr;

		residuals[0] = (lp - lpj).dot(ljm);

		if (jacobians) 
		{
			if (jacobians[0])
			{
				// jacobians[0][0] = 0;
				// jacobians[0][1] = 0;
				// jacobians[0][2] = -ljm(0) * (q_last_curr * cp)(1) + ljm(1) * (q_last_curr * cp)(0);
				// jacobians[0][3] = 0;



				Eigen::Matrix3d ror_mat = q_last_curr.toRotationMatrix();
                Eigen::Matrix3d point_mat = mathutils::SkewSymmetric(cp);
				
				Eigen::Matrix3d right_dev = -ror_mat * point_mat;

				Eigen::Matrix<double, 1, 3> result = ljm.transpose() * right_dev;


				jacobians[0][0] = 0;
				jacobians[0][1] = 0;
				jacobians[0][2] = result(0, 2);
				jacobians[0][3] = 0;
			}
			if (jacobians[1]) 
			{
				jacobians[1][0] = ljm(0);
				jacobians[1][1] = ljm(1);
				jacobians[1][2] = 0;
			}
		}

		return true;
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m, ljm_norm;
	double s;
};


class LidarPlaneFactor_z_trans : public ceres::SizedCostFunction<1, 4, 3>
{
    public:

	LidarPlaneFactor_z_trans(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {


		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lpj{last_point_j.x(), last_point_j.y(), last_point_j.z()};
		Eigen::Matrix<double, 3, 1> ljm{ljm_norm.x(), ljm_norm.y(), ljm_norm.z()};

        Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};

		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

		Eigen::Matrix<double, 3, 1> lp = q_last_curr * cp + t_last_curr;

		residuals[0] = (lp - lpj).dot(ljm);

		if (jacobians) 
		{
			if (jacobians[0])
			{
				// jacobians[0][0] = 0;
				// jacobians[0][1] = 0;
				// jacobians[0][2] = 0;
				// jacobians[0][3] = 0;


				Eigen::Matrix3d ror_mat = q_last_curr.toRotationMatrix();
                Eigen::Matrix3d point_mat = mathutils::SkewSymmetric(cp);
				
				Eigen::Matrix3d right_dev = -ror_mat * point_mat;

				Eigen::Matrix<double, 1, 3> result = ljm.transpose() * right_dev;


				jacobians[0][0] = 0;
				jacobians[0][1] = 0;
				jacobians[0][2] = 0;
				jacobians[0][3] = 0;

			}
			if (jacobians[1]) 
			{
				jacobians[1][0] = 0;
				jacobians[1][1] = 0;
				jacobians[1][2] = ljm(2);
			}
		}

		return true;
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m, ljm_norm;
	double s;
};


class LidarPlaneFactor_x_rot : public ceres::SizedCostFunction<1, 4, 3>
{
    public:

	LidarPlaneFactor_x_rot(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {


		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lpj{last_point_j.x(), last_point_j.y(), last_point_j.z()};
		Eigen::Matrix<double, 3, 1> ljm{ljm_norm.x(), ljm_norm.y(), ljm_norm.z()};

        Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};

		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

		Eigen::Matrix<double, 3, 1> lp = q_last_curr * cp + t_last_curr;

		residuals[0] = (lp - lpj).dot(ljm);

		if (jacobians) 
		{
			if (jacobians[0])
			{
				// jacobians[0][0] = -ljm(1) * (q_last_curr * cp)(2) + ljm(2) * (q_last_curr * cp)(1);
				// jacobians[0][1] = 0;
				// jacobians[0][2] = 0;
				// jacobians[0][3] = 0;



				Eigen::Matrix3d ror_mat = q_last_curr.toRotationMatrix();
                Eigen::Matrix3d point_mat = mathutils::SkewSymmetric(cp);
				
				Eigen::Matrix3d right_dev = -ror_mat * point_mat;

				Eigen::Matrix<double, 1, 3> result = ljm.transpose() * right_dev;


				jacobians[0][0] = result(0, 0);
				jacobians[0][1] = 0;
				jacobians[0][2] = 0;
				jacobians[0][3] = 0;
			}
			if (jacobians[1]) 
			{
				jacobians[1][0] = 0;
				jacobians[1][1] = 0;
				jacobians[1][2] = 0;
			}
		}

		return true;
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m, ljm_norm;
	double s;
};


class LidarPlaneFactor_y_rot : public ceres::SizedCostFunction<1, 4, 3>
{
    public:

	LidarPlaneFactor_y_rot(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {


		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lpj{last_point_j.x(), last_point_j.y(), last_point_j.z()};
		Eigen::Matrix<double, 3, 1> ljm{ljm_norm.x(), ljm_norm.y(), ljm_norm.z()};

        Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};

		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

		Eigen::Matrix<double, 3, 1> lp = q_last_curr * cp + t_last_curr;

		residuals[0] = (lp - lpj).dot(ljm);

		if (jacobians) 
		{
			if (jacobians[0])
			{
				// jacobians[0][0] = 0;
				// jacobians[0][1] = ljm(0) * (q_last_curr * cp)(2) - ljm(2) * (q_last_curr * cp)(0);
				// jacobians[0][2] = 0;
				// jacobians[0][3] = 0;


				Eigen::Matrix3d ror_mat = q_last_curr.toRotationMatrix();
                Eigen::Matrix3d point_mat = mathutils::SkewSymmetric(cp);
				
				Eigen::Matrix3d right_dev = -ror_mat * point_mat;

				Eigen::Matrix<double, 1, 3> result = ljm.transpose() * right_dev;


				jacobians[0][0] = 0;
				jacobians[0][1] = result(0, 1);
				jacobians[0][2] = 0;
				jacobians[0][3] = 0;

			}
			if (jacobians[1]) 
			{
				jacobians[1][0] = 0;
				jacobians[1][1] = 0;
				jacobians[1][2] = 0;
			}
		}

		return true;
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m, ljm_norm;
	double s;
};




class LidarPlaneFactor_xyz_rot_xyz_trans : public ceres::SizedCostFunction<1, 4, 3>
{
    public:

	LidarPlaneFactor_xyz_rot_xyz_trans(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const {


		Eigen::Matrix<double, 3, 1> cp{curr_point.x(), curr_point.y(), curr_point.z()};
		Eigen::Matrix<double, 3, 1> lpj{last_point_j.x(), last_point_j.y(), last_point_j.z()};
		Eigen::Matrix<double, 3, 1> ljm{ljm_norm.x(), ljm_norm.y(), ljm_norm.z()};

        Eigen::Quaterniond q_last_curr{parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};

		Eigen::Matrix<double, 3, 1> t_last_curr{parameters[1][0], parameters[1][1], parameters[1][2]};

		Eigen::Matrix<double, 3, 1> lp = q_last_curr * cp + t_last_curr;

		residuals[0] = (lp - lpj).dot(ljm);

		if (jacobians) 
		{
			if (jacobians[0])
			{
				// jacobians[0][0] = -ljm(1) * (q_last_curr * cp)(2) + ljm(2) * (q_last_curr * cp)(1);
				// jacobians[0][1] = ljm(0) * (q_last_curr * cp)(2) - ljm(2) * (q_last_curr * cp)(0);
				// jacobians[0][2] = -ljm(0) * (q_last_curr * cp)(1) + ljm(1) * (q_last_curr * cp)(0);
				// jacobians[0][3] = 0;


				Eigen::Matrix3d ror_mat = q_last_curr.toRotationMatrix();
                Eigen::Matrix3d point_mat = mathutils::SkewSymmetric(cp);
				
				Eigen::Matrix3d right_dev = -ror_mat * point_mat;

				Eigen::Matrix<double, 1, 3> result = ljm.transpose() * right_dev;


				jacobians[0][0] = result(0, 0);
				jacobians[0][1] = result(0, 1);
				jacobians[0][2] = result(0, 2);
				jacobians[0][3] = 0;				
			}
			if (jacobians[1]) 
			{
				jacobians[1][0] = ljm(0);
				jacobians[1][1] = ljm(1);
				jacobians[1][2] = ljm(2);
			}
		}

		return true;
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m, ljm_norm;
	double s;
};




struct LidarPlaneFactor
{
	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		//Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
		//Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		residual[0] = (lp - lpj).dot(ljm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneFactor, 1, 4, 3>(
			new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;
};



struct LidarPlane_Norm_Factor
{
	LidarPlane_Norm_Factor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, Eigen::Vector3d curr_point_norm_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), curr_point_norm(curr_point_norm_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		//Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
		//Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;
		//residual[0] =  (lp - lpj).dot(curr_point_norm);
        //residual[0] = sqrt((fabs(ljm_norm.transpose().dot(curr_point_norm)))) * (lp - lpj).dot(ljm);
		//residual[0] = (fabs(ljm_norm.transpose().dot(curr_point_norm))) * (lp - lpj).dot(ljm);
		residual[0] = (fabs(ljm_norm.dot(curr_point_norm))) * (lp - lpj).dot(ljm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const Eigen::Vector3d curr_point_norm_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlane_Norm_Factor, 1, 4, 3>(
			new LidarPlane_Norm_Factor(curr_point_, last_point_j_, last_point_l_, last_point_m_, curr_point_norm_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m, curr_point_norm;
	Eigen::Vector3d ljm_norm;
	double s;
};




struct LidarPlaneNormFactor
{

	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
														 negative_OA_dot_norm(negative_OA_dot_norm_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactor, 1, 4, 3>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};


struct LidarDistanceFactor
{

	LidarDistanceFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_) 
						: curr_point(curr_point_), closed_point(closed_point_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;


		residual[0] = point_w.x() - T(closed_point.x());
		residual[1] = point_w.y() - T(closed_point.y());
		residual[2] = point_w.z() - T(closed_point.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactor, 3, 4, 3>(
			new LidarDistanceFactor(curr_point_, closed_point_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
};


struct LidarDistanceFactorLast
{

	LidarDistanceFactorLast(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_) 
						: curr_point(curr_point_), closed_point(closed_point_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		//point_w = q_w_curr * cp + t_w_curr;
        point_w = q_w_curr.inverse() * (cp - t_w_curr);

		residual[0] = point_w.x() - T(closed_point.x());
		residual[1] = point_w.y() - T(closed_point.y());
		residual[2] = point_w.z() - T(closed_point.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactorLast, 3, 4, 3>(
			new LidarDistanceFactorLast(curr_point_, closed_point_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
};




struct EndBackFactor
{
	EndBackFactor(Eigen::Quaterniond pretonextq_, Eigen::Matrix<double, 3, 1> pretonextt_)
		          : pretonextq(pretonextq_), pretonextt(pretonextt_){}
template <typename T>
	bool operator()(const T *q_0, const T *t_0, const T *q_1, const T *t_1, T *residual) const
	{
		Eigen::Quaternion<T> q_pre{q_0[3], q_0[0], q_0[1], q_0[2]};
		Eigen::Quaternion<T> q_next{q_1[3], q_1[0], q_1[1], q_1[2]};

		Eigen::Matrix<T, 3, 1> t_pre{t_0[0],t_0[1], t_0[2]};
		Eigen::Matrix<T, 3, 1> t_next{t_1[0], t_1[1], t_1[2]};

		Eigen::Quaternion<T> pretonext_q = q_next.inverse() * q_pre;
		Eigen::Matrix<T, 3, 1> pretonext_t = q_next.inverse() * (t_pre - t_next);

        residual[0] = pretonext_q.x() - T(pretonextq.x());
		residual[1] = pretonext_q.y() - T(pretonextq.y());
		residual[2] = pretonext_q.z() - T(pretonextq.z());
		residual[3] = pretonext_t(0,0) - T(pretonextt(0,0));
		residual[4] = pretonext_t(1,0) - T(pretonextt(1,0));
		residual[5] = pretonext_t(2,0) - T(pretonextt(2,0));
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Quaterniond pretonextq_, const Eigen::Matrix<double, 3, 1> pretonextt_)
	{
		return (new ceres::AutoDiffCostFunction<
				EndBackFactor, 6, 4, 3, 4, 3>(
			new EndBackFactor(pretonextq_, pretonextt_)));
	}
	Eigen::Quaterniond pretonextq;
	Eigen::Matrix<double, 3, 1> pretonextt;


};


struct EndBackFirstFreamFactor
{
	EndBackFirstFreamFactor(Eigen::Quaterniond pretonextq_, Eigen::Matrix<double, 3, 1> pretonextt_, Eigen::Quaterniond firstfreamq_, Eigen::Matrix<double, 3, 1> firstfreamt_)
		          : pretonextq(pretonextq_), pretonextt(pretonextt_), firstfreamq(firstfreamq_), firstfreamt(firstfreamt_){}
    template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion <T> q_next{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_next{t[0],t[1], t[2]};

		Eigen::Quaternion<T> pretonext_q = q_next.inverse() * firstfreamq.cast<T>();
		Eigen::Matrix<T, 3, 1> pretonext_t = q_next.inverse() * (firstfreamt.cast<T>() - t_next);

        residual[0] = pretonext_q.x() - T(pretonextq.x());
		residual[1] = pretonext_q.y() - T(pretonextq.y());
		residual[2] = pretonext_q.z() - T(pretonextq.z());
		residual[3] = pretonext_t(0,0) - T(pretonextt(0,0));
		residual[4] = pretonext_t(1,0) - T(pretonextt(1,0));
		residual[5] = pretonext_t(2,0) - T(pretonextt(2,0));
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Quaterniond pretonextq_, const Eigen::Matrix<double, 3, 1> pretonextt_, const Eigen::Quaterniond fistfreamq_, const Eigen::Matrix<double, 3, 1> firstfreamt_)
	{
		return (new ceres::AutoDiffCostFunction<
				EndBackFirstFreamFactor, 6, 4, 3>(
			new EndBackFirstFreamFactor(pretonextq_, pretonextt_, fistfreamq_, firstfreamt_)));
	}
	Eigen::Quaterniond pretonextq;
	Eigen::Matrix<double, 3, 1> pretonextt;

	Eigen::Quaterniond firstfreamq;
    Eigen::Matrix<double, 3, 1> firstfreamt;

};