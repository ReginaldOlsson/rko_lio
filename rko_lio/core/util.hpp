/*
 * MIT License
 *
 * Copyright (c) 2025 Meher V.R. Malladi.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once
#include <Eigen/Core>
#include <chrono>
#include <cstdint>
#include <memory>
#include <sophus/se3.hpp>
#include <vector>

namespace Eigen {
using Matrix3_6d = Matrix<double, 3, 6>;
using Vector6d = Matrix<double, 6, 1>;
using Matrix6d = Matrix<double, 6, 6>;
using Matrix12d = Matrix<double, 12, 12>;
} // namespace Eigen

namespace rko_lio::core {
// aliases
using Vector3dVector = std::vector<Eigen::Vector3d>;
using Secondsd = std::chrono::duration<double>;
using TimestampVector = std::vector<Secondsd>;

// constants and util funcs
constexpr double square(double x) { return x * x; }
constexpr double GRAVITY_MAG = 9.8107;
inline Eigen::Vector3d gravity() { return {0, 0, -GRAVITY_MAG}; }

// data structs
struct State {
  Secondsd time{0};
  Sophus::SE3d pose;
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
};

struct ImuBias {
  Eigen::Vector3d accelerometer = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyroscope = Eigen::Vector3d::Zero();
};

struct ImuControl {
  Secondsd time{0};
  Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
};

struct Timestamps {
  Secondsd min;
  Secondsd max;
  TimestampVector times;
};

struct LidarFrame {
  Timestamps timestamps;
  Vector3dVector points;
};

/** Accumulated IMU statistics over the interval between consecutive LiDAR scans. */
struct IntervalStats {
  /** Number of IMU samples accumulated during this interval. */
  int imu_count = 0;

  /** Sum of unbiased angular velocity samples. */
  Eigen::Vector3d angular_velocity_sum = Eigen::Vector3d::Zero();

  /** Sum of gravity-compensated body-frame accelerations. */
  Eigen::Vector3d body_acceleration_sum = Eigen::Vector3d::Zero();

  /** Sum of unbiased raw IMU accelerations. */
  Eigen::Vector3d imu_acceleration_sum = Eigen::Vector3d::Zero();

  /** Mean magnitude of raw IMU acceleration over the interval. */
  double imu_accel_mag_mean = 0;

  /** Variance accumulator for acceleration magnitude using Welford’s method. */
  double welford_sum_of_squares = 0;

  /**
   * Update accumulated statistics with a new IMU measurement.
   * @param unbiased_ang_vel Unbiased angular velocity.
   * @param uncompensated_unbiased_accel Uncompensated, unbiased acceleration.
   * @param compensated_accel Gravity-compensated acceleration.
   */
  void update(const Eigen::Vector3d& unbiased_ang_vel,
              const Eigen::Vector3d& uncompensated_unbiased_accel,
              const Eigen::Vector3d& compensated_accel) {
    ++imu_count;
    angular_velocity_sum += unbiased_ang_vel;
    imu_acceleration_sum += uncompensated_unbiased_accel;

    const double previous_mean = imu_accel_mag_mean;
    const double accel_norm = uncompensated_unbiased_accel.norm();

    imu_accel_mag_mean += (accel_norm - previous_mean) / imu_count;
    welford_sum_of_squares += (accel_norm - previous_mean) * (accel_norm - imu_accel_mag_mean);

    body_acceleration_sum += compensated_accel;
  }

  /** Reset all accumulated statistics to zero. */
  void reset() {
    imu_count = 0;
    angular_velocity_sum.setZero();
    body_acceleration_sum.setZero();
    imu_acceleration_sum.setZero();
    imu_accel_mag_mean = 0;
    welford_sum_of_squares = 0;
  }
};

/** Detail: return type for the acceleration Kalman filter containing gravity and variance estimates. */
struct AccelInfo {
  /** Variance of the raw imu acceleration magnitude. */
  double accel_mag_variance;
  Eigen::Vector3d local_gravity_estimate;
};

/**
 * Pinhole camera intrinsics for a rectified image.
 *
 * Distortion is expected to be removed in the ROS layer using the
 * subscribed CameraInfo (D, K, R, P matrices); the core consumes only the
 * rectified pinhole model.
 */
struct PinholeIntrinsics {
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  int width = 0;
  int height = 0;
};

/**
 * A camera measurement, preprocessed in the ROS (or Python) layer, ready for
 * use by the camera edge-alignment residual.
 *
 * `dt_image` carries the unsigned L2 distance transform of the rectified
 * image's Canny edges, in pixels, as a row-major contiguous float buffer of
 * size `rows * cols`. Shared ownership keeps the buffer alive for as long
 * as any consumer holds the frame.
 *
 * The core never touches OpenCV: the ROS layer is responsible for running
 * Canny + cv::distanceTransform and producing the buffer.
 */
struct CameraFrame {
  Secondsd time{0};
  PinholeIntrinsics intrinsics;
  std::shared_ptr<const std::vector<float>> dt_image;
  int rows = 0;
  int cols = 0;
};

/**
 * Per-voxel dynamic statistics maintained as a side table parallel to the
 * sparse voxel grid. All fields default to a "no information" state so newly
 * observed voxels are treated as static until the EMA settles. Only updated
 * when `LIO::Config::dynamic_segmentation_enabled` is true.
 */
struct VoxelDynStats {
  /** EMA of the mean correspondence residual to this voxel, in meters. */
  float resid_ema = 0.0f;
  /** EMA of the dynamic indicator `1[r > tau_dynamic || rejected]`. */
  float dyn_score = 0.0f;
  /** Number of scans in which this voxel participated as a correspondence. */
  uint16_t hit_count = 0;
  /** Reserved for future visibility-based dynamic carving. */
  uint16_t miss_count = 0;
  /** Scan counter at the most recent telemetry update for this voxel. */
  uint32_t last_seen_scan = 0;
};
}; // namespace rko_lio::core
