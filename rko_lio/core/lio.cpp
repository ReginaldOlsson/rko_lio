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

#include "lio.hpp"
#include "camera_residual.hpp"
#include "preprocess_scan.hpp"
#include "profiler.hpp"
#include "util.hpp"
// other
#include <sophus/se3.hpp>
// tbb
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
// stl
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
constexpr double EPSILON = 1e-8;
constexpr auto EPSILON_TIME = std::chrono::nanoseconds(10);
using namespace rko_lio::core;

inline void transform_points(const Sophus::SE3d& T, Vector3dVector& points) {
  std::transform(points.begin(), points.end(), points.begin(), [&](const auto& point) { return T * point; });
}

/**
 * Optional read-only context used inside ICP to apply a per-correspondence
 * weight `w_i = exp(-decay_k * dyn_score)` when dynamic-point segmentation
 * is enabled. When the context pointer is null, the ICP solver behaves
 * identically to the unmodified core.
 *
 * Reading from the side table during the parallel reduce is safe because
 * the table is only written once per scan, sequentially, after the ICP
 * loop exits.
 */
struct DynWeightingContext {
  const std::unordered_map<Bonxai::CoordT, VoxelDynStats>* voxel_dyn = nullptr;
  double weight_decay_k = 0.0;
  int min_hits_to_trust = 0;
};

/** w_i = exp(-decay_k * dyn_score) once a voxel has enough hits to be trusted. */
inline double dyn_weight_for_voxel(const DynWeightingContext* ctx, const Bonxai::CoordT& key) {
  if (ctx == nullptr || ctx->voxel_dyn == nullptr) {
    return 1.0;
  }
  const auto it = ctx->voxel_dyn->find(key);
  if (it == ctx->voxel_dyn->end()) {
    return 1.0;
  }
  if (static_cast<int>(it->second.hit_count) < ctx->min_hits_to_trust) {
    return 1.0;
  }
  return std::exp(-ctx->weight_decay_k * static_cast<double>(it->second.dyn_score));
}

LinearSystem build_icp_linear_system(const Sophus::SE3d& current_pose,
                                     const rko_lio::core::Vector3dVector& frame,
                                     const rko_lio::core::SparseVoxelGrid& voxel_map,
                                     const double& max_correspondance_distance,
                                     const DynWeightingContext* dyn_ctx = nullptr) {
  auto linear_system_reduce = [](LinearSystem lhs, const LinearSystem& rhs) {
    auto& [lhs_H, lhs_b, lhs_chi] = lhs;
    const auto& [rhs_H, rhs_b, rhs_chi] = rhs;
    lhs_H += rhs_H;
    lhs_b += rhs_b;
    lhs_chi += rhs_chi;
    return lhs;
  };

  auto linear_system_for_one_point = [](const Eigen::Vector3d& source, const Eigen::Vector3d& target, double weight) {
    Eigen::Matrix3_6d J_r;
    J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source);
    const Eigen::Vector3d residual = source - target;
    return LinearSystem(weight * J_r.transpose() * J_r,      // w * JTJ
                        weight * J_r.transpose() * residual, // w * JTr
                        weight * residual.squaredNorm());    // w * chi (for diagnostics)
  };

  // The only parallel part
  using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
  std::atomic<int> correspondances_counter = 0;
  const auto& [H_icp, b_icp, chi_icp] = tbb::parallel_reduce(
      // Range
      tbb::blocked_range<points_iterator>{frame.cbegin(), frame.cend()},
      // Identity
      LinearSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero(), 0.0),
      // 1st Lambda: Parallel computation
      [&](const tbb::blocked_range<points_iterator>& r, LinearSystem J) -> LinearSystem {
        return std::transform_reduce(r.begin(), r.end(), J, linear_system_reduce, [&](const auto& point) {
          // Compute data association and linear system
          const Eigen::Vector3d transformed_point = current_pose * point;
          const auto& [closest_neighbor, distance, voxel_key] = voxel_map.GetClosestNeighbor(transformed_point);
          if (distance < max_correspondance_distance) {
            correspondances_counter++;
            const double w = dyn_weight_for_voxel(dyn_ctx, voxel_key);
            return linear_system_for_one_point(transformed_point, closest_neighbor, w);
          }
          // TODO (meher): additional 0 add flops, which may hurt single threaded perf slightly
          return LinearSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero(), 0.0);
        });
      },
      // 2nd Lambda: Parallel reduction of the private Jacobians
      linear_system_reduce);

  if (correspondances_counter == 0) {
    throw std::runtime_error("Number of correspondences are 0.");
  }

  return {H_icp / correspondances_counter, b_icp / correspondances_counter, 0.5 * chi_icp};
}

/**
 * Per-correspondence telemetry record used by the dynamic-segmentation
 * post-pass. Collected once per scan, after the ICP loop has converged,
 * so the EMA in `voxel_dyn` is updated using residuals at the final pose
 * estimate.
 */
struct CorrespondenceTelemetry {
  Bonxai::CoordT voxel_key{0, 0, 0};
  float residual_m = 0.f;
  bool accepted = false; // matched within `max_correspondance_distance`
  bool valid = false;    // false when the scan point has no neighbor at all
};

std::vector<CorrespondenceTelemetry> collect_icp_telemetry(const Sophus::SE3d& final_pose,
                                                            const Vector3dVector& frame,
                                                            const SparseVoxelGrid& voxel_map,
                                                            double max_correspondance_distance) {
  std::vector<CorrespondenceTelemetry> telemetry(frame.size());

  tbb::parallel_for(tbb::blocked_range<size_t>(0, frame.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      const Eigen::Vector3d transformed_point = final_pose * frame[i];
      const auto& [closest_neighbor, distance, voxel_key] = voxel_map.GetClosestNeighbor(transformed_point);
      if (distance == std::numeric_limits<double>::max()) {
        telemetry[i] = {.voxel_key = {0, 0, 0}, .residual_m = 0.f, .accepted = false, .valid = false};
      } else {
        telemetry[i] = {.voxel_key = voxel_key,
                        .residual_m = static_cast<float>(distance),
                        .accepted = distance < max_correspondance_distance,
                        .valid = true};
      }
    }
  });

  return telemetry;
}

LinearSystem build_orientation_linear_system(const Sophus::SE3d& current_pose,
                                             const Eigen::Vector3d& local_gravity_estimate) {
  const Sophus::SO3d& current_rotation = current_pose.so3();
  const Eigen::Vector3d predicted_gravity =
      current_rotation.inverse() * (-1 * gravity()); // points upwards, same as local_gravity_estimate
  const Eigen::Vector3d residual = predicted_gravity - local_gravity_estimate;

  Eigen::Matrix3_6d J_ori = Eigen::Matrix3_6d::Zero();
  J_ori.block<3, 3>(0, 3) = current_rotation.inverse().matrix() * Sophus::SO3d::hat(-1 * gravity()).matrix();

  return LinearSystem{J_ori.transpose() * J_ori, J_ori.transpose() * residual, 0.5 * residual.squaredNorm()};
}

Sophus::SE3d icp(const Vector3dVector& frame,
                 const SparseVoxelGrid& voxel_map,
                 const Sophus::SE3d& initial_guess,
                 const LIO::Config& config,
                 const std::optional<AccelInfo>& optional_accel_info,
                 const DynWeightingContext* dyn_ctx,
                 const CameraResidualContext* camera_ctx) {
  // in case config disables it, or we don't have valid IMU information for this icp loop, beta is -1
  const double beta = (config.min_beta > 0 && optional_accel_info.has_value())
                          ? (config.min_beta * (1 + optional_accel_info->accel_mag_variance))
                          : -1;

  const bool camera_active = camera_ctx != nullptr && camera_ctx->frame != nullptr &&
                             camera_ctx->extrinsic_cam_to_base != nullptr && camera_ctx->visible != nullptr;

  Sophus::SE3d current_pose = initial_guess;

  for (size_t i = 0; i < config.max_iterations; ++i) {
    const auto& [H, b, chi] = std::invoke([&]() -> LinearSystem {
      auto [H_sum, b_sum, chi_sum] =
          build_icp_linear_system(current_pose, frame, voxel_map, config.max_correspondance_distance, dyn_ctx);
      if (beta >= 0) {
        const auto& [H_ori, b_ori, chi_ori] =
            build_orientation_linear_system(current_pose, optional_accel_info->local_gravity_estimate);
        H_sum += H_ori / beta;
        b_sum += b_ori / beta;
        chi_sum += chi_ori / beta;
      }
      if (camera_active) {
        const auto& [H_cam, b_cam, chi_cam] = build_camera_edge_linear_system(
            current_pose, *camera_ctx->extrinsic_cam_to_base, *camera_ctx->frame, *camera_ctx->visible, config);
        H_sum += H_cam;
        b_sum += b_cam;
        chi_sum += chi_cam;
      }
      return {H_sum, b_sum, chi_sum};
    });

    const Eigen::Vector6d dx = H.ldlt().solve(-b);
    current_pose = Sophus::SE3d::exp(dx) * current_pose;

    if (dx.norm() < config.convergence_criterion || i == (config.max_iterations - 1)) {
      // TODO: proper debug logging
      // std::cout << "iter " << i << ", beta: " << beta << ", chi: " << chi << ", num_assoc: " <<
      // correspondences.size() << "\n";
      break;
    }
  }
  return current_pose;
}

/**
 * Update the per-voxel dynamic-segmentation side table using the telemetry
 * collected at the converged pose. Mutates `voxel_dyn` in place.
 *
 * The EMA mixing factor is `dyn_ema_alpha`. The dynamic indicator fires when
 * the residual exceeds `dyn_tau_dynamic_m`, or when the correspondence was
 * rejected by the `max_correspondance_distance` gate (which also counts as
 * "dynamic-suspicious").
 */
void update_voxel_dyn_stats(std::unordered_map<Bonxai::CoordT, VoxelDynStats>& voxel_dyn,
                            const std::vector<CorrespondenceTelemetry>& telemetry,
                            const LIO::Config& config,
                            uint32_t scan_counter) {
  const double alpha = config.dyn_ema_alpha;
  const double one_minus_alpha = 1.0 - alpha;
  const double tau_dyn = config.dyn_tau_dynamic_m;

  for (const auto& t : telemetry) {
    if (!t.valid) {
      continue;
    }
    auto& stats = voxel_dyn[t.voxel_key];
    const bool is_dynamic_evidence = !t.accepted || (t.residual_m > tau_dyn);
    stats.resid_ema =
        static_cast<float>(one_minus_alpha * static_cast<double>(stats.resid_ema) + alpha * t.residual_m);
    stats.dyn_score = static_cast<float>(one_minus_alpha * static_cast<double>(stats.dyn_score) +
                                         alpha * (is_dynamic_evidence ? 1.0 : 0.0));
    if (stats.hit_count < std::numeric_limits<uint16_t>::max()) {
      ++stats.hit_count;
    }
    stats.last_seen_scan = scan_counter;
  }
}

/**
 * Build an `accept_mask` for `SparseVoxelGrid::Update`: scan points whose
 * target voxel has a trusted `dyn_score > dyn_skip_map_score` are excluded
 * so dynamic returns do not pollute the map.
 *
 * Operates on the *map-update* point set (not the ICP keypoint set), since
 * those are the points actually added to the map. The voxel each point
 * would land in is derived from `pose * point`. Untrusted voxels
 * (`hit_count` below the gate) default to "accept" so newly observed
 * surfaces are not blocked from joining the map.
 */
std::vector<uint8_t> build_map_accept_mask(const Vector3dVector& map_update_points,
                                           const Sophus::SE3d& pose,
                                           const SparseVoxelGrid& voxel_map,
                                           const std::unordered_map<Bonxai::CoordT, VoxelDynStats>& voxel_dyn,
                                           const LIO::Config& config) {
  std::vector<uint8_t> mask(map_update_points.size(), 1);
  const double gate = config.dyn_skip_map_score;
  const int min_hits = config.dyn_min_hits_to_trust;
  for (size_t i = 0; i < map_update_points.size(); ++i) {
    const Bonxai::CoordT key = voxel_map.PosToCoord(pose * map_update_points[i]);
    const auto it = voxel_dyn.find(key);
    if (it == voxel_dyn.end()) {
      continue;
    }
    if (static_cast<int>(it->second.hit_count) < min_hits) {
      continue;
    }
    if (static_cast<double>(it->second.dyn_score) > gate) {
      mask[i] = 0;
    }
  }
  return mask;
}

inline Sophus::SO3d align_accel_to_z_world(const Eigen::Vector3d& accel) {
  //  unobservable in the gravity direction, and the z in R.log() will always be 0
  const Eigen::Vector3d z_world = {0.0, 0.0, 1.0};
  const Eigen::Quaterniond quat_accel = Eigen::Quaterniond::FromTwoVectors(accel, z_world);
  return Sophus::SO3d(quat_accel);
}
} // namespace

// ==========================
//   actual LIO class stuff
// ==========================

namespace rko_lio::core {

// ==========================
//          private
// ==========================

void LIO::initialize(const Secondsd lidar_time) {
  if (interval_stats.imu_count == 0) {
    std::cerr << "[WARNING] Cannot initialize. No imu measurements received.\n";
    // lidar_state.time has the time from the previous lidar, which we didn't log if init_phase was on
    poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
    _initialized = true;
    return;
  }

  const Eigen::Vector3d avg_accel = interval_stats.imu_acceleration_sum / interval_stats.imu_count;
  const Eigen::Vector3d avg_gyro = interval_stats.angular_velocity_sum / interval_stats.imu_count;

  _imu_local_rotation = align_accel_to_z_world(avg_accel);
  _imu_local_rotation_time = lidar_time;
  lidar_state.pose.so3() = _imu_local_rotation;

  // lidar_state.time has the time from the previous lidar, which we didn't log if init_phase was on
  poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);

  // the pose for the current time gets logged at the end of register_scan in the typical fashion
  lidar_state.time = lidar_time;

  const Eigen::Vector3d local_gravity = _imu_local_rotation.inverse() * gravity();
  imu_bias.accelerometer = avg_accel + local_gravity;
  imu_bias.gyroscope = avg_gyro;

  _initialized = true;
  std::cout << "[INFO] Odometry map frame initialized using " << interval_stats.imu_count
            << " IMU measurements. Estimated initial rotation [se(3)] is " << _imu_local_rotation.log().transpose()
            << "\n";
  std::cout << "[INFO] Estimated accel bias: " << imu_bias.accelerometer.transpose()
            << ", gyro bias: " << imu_bias.gyroscope.transpose() << "\n";
}

// use the acceleration kalman filter to compute the two values we need for ori. reg.
std::optional<AccelInfo> LIO::get_accel_info(const Sophus::SO3d& rotation_estimate, const Secondsd& time) {
  if (interval_stats.imu_count <= 1) {
    std::cerr << "[WARNING] " << interval_stats.imu_count
              << " IMU message(s) in interval between two lidar scans. Cannot compute "
                 "acceleration statistics for orientation regularisation. Please check your data and its "
                 "timestamping as likely there should not be so few IMU measurements between two LiDAR scans.\n";
    return std::nullopt;
  }

  const Eigen::Vector3d avg_imu_accel = interval_stats.imu_acceleration_sum / interval_stats.imu_count;
  const double accel_mag_variance = interval_stats.welford_sum_of_squares / (interval_stats.imu_count - 1);
  const double dt = (time - lidar_state.time).count();

  const Eigen::Vector3d& body_accel_measurement = avg_imu_accel + rotation_estimate.inverse() * gravity();

  const double max_acceleration_change = config.max_expected_jerk * dt;
  // assume [j, -j] range for uniform dist. on jerk. variance is (2j)^2 / 12 = j^2/3. multiply by dt^2 for accel
  const Eigen::Matrix3d process_noise = square(max_acceleration_change) / 3 * Eigen::Matrix3d::Identity();
  body_acceleration_covariance += process_noise;

  // isotropic accel mag variance
  const Eigen::Matrix3d measurement_noise = accel_mag_variance / 3 * Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d S = body_acceleration_covariance + measurement_noise;
  const Eigen::Matrix3d kalman_gain = body_acceleration_covariance * S.inverse();

  const Eigen::Vector3d innovation = kalman_gain * (body_accel_measurement - mean_body_acceleration);
  mean_body_acceleration += innovation;
  body_acceleration_covariance -= kalman_gain * body_acceleration_covariance;

  const Eigen::Vector3d local_gravity_estimate = avg_imu_accel - mean_body_acceleration; // points upwards

  return AccelInfo{.accel_mag_variance = accel_mag_variance, .local_gravity_estimate = local_gravity_estimate};
}

// ==========================
//          public
// ==========================

// ============================ imu ===============================

void LIO::add_imu_measurement(const ImuControl& base_imu) {
  if (lidar_state.time < EPSILON_TIME) {
    static bool warning_skip_till_first_lidar = false;
    if (!warning_skip_till_first_lidar) {
      std::cerr << "[WARNING - ONCE] Skipping IMU, waiting for first LiDAR message.\n";
      warning_skip_till_first_lidar = true;
    }
    _last_real_imu_time = base_imu.time;
    _last_real_base_imu_ang_vel = base_imu.angular_velocity;
    return;
  }

  if (_imu_local_rotation_time < EPSILON_TIME) {
    _imu_local_rotation_time = lidar_state.time;
  }

  const double dt = (base_imu.time - _imu_local_rotation_time).count();

  if (dt < 0.0) {
    // messages are out of sync. thats a problem, since we integrate gyro from last lidar time onwards
    std::cerr << "[WARNING] Received IMU message from the past. Can result in errors.\n";
    // maybe skip this imu reading?
  }

  const Eigen::Vector3d unbiased_ang_vel = base_imu.angular_velocity - imu_bias.gyroscope;
  const Eigen::Vector3d unbiased_accel = base_imu.acceleration - imu_bias.accelerometer;

  _imu_local_rotation = _imu_local_rotation * Sophus::SO3d::exp(unbiased_ang_vel * dt);
  _imu_local_rotation_time = base_imu.time;

  const Eigen::Vector3d local_gravity = _imu_local_rotation.inverse() * gravity();
  const Eigen::Vector3d compensated_accel = unbiased_accel + local_gravity;

  interval_stats.update(unbiased_ang_vel, unbiased_accel, compensated_accel);

  _last_real_imu_time = base_imu.time;
  _last_real_base_imu_ang_vel = base_imu.angular_velocity;
}

void LIO::add_imu_measurement(const Sophus::SE3d& extrinsic_imu2base, const ImuControl& raw_imu) {
  if (extrinsic_imu2base.log().norm() < EPSILON) {
    add_imu_measurement(raw_imu);
    return;
  }

  if (_last_real_imu_time < EPSILON_TIME) {
    // skip IMU message as we need a previous imu time for extrinsic compensation
    _last_real_imu_time = raw_imu.time;
    return;
  }

  // accounting for the transport-rate
  ImuControl base_imu = raw_imu;
  const Sophus::SO3d& extrinsic_rotation = extrinsic_imu2base.so3();
  base_imu.angular_velocity = extrinsic_rotation * raw_imu.angular_velocity;

  const Eigen::Vector3d& lever_arm = -1 * extrinsic_imu2base.translation();
  const Secondsd dt = raw_imu.time - _last_real_imu_time;

  const Eigen::Vector3d angular_acceleration = std::invoke([&]() -> Eigen::Vector3d {
    if (std::chrono::abs(dt) < Secondsd(1.0 / 5000.0)) {
      // if dt is less than the equivalent of a 5000 Hz imu, assuming zero ang accel,
      // causes numerical issues otherwise
      static bool warning_imu_too_close = false;
      if (!warning_imu_too_close) {
        std::cerr << "[WARNING - ONCE] Received IMU message with a very short delta to previous IMU message. Ignoring "
                     "all such messages.\n";
        warning_imu_too_close = true;
      }
      return Eigen::Vector3d::Zero();
    } else {
      const Eigen::Vector3d angular_acceleration =
          (base_imu.angular_velocity - _last_real_base_imu_ang_vel) / dt.count();
      return angular_acceleration;
    }
  });

  base_imu.acceleration = extrinsic_rotation * raw_imu.acceleration + angular_acceleration.cross(lever_arm) +
                          base_imu.angular_velocity.cross(base_imu.angular_velocity.cross(lever_arm));

  this->add_imu_measurement(base_imu);
}

// ============================ lidar ===============================

Vector3dVector LIO::register_scan(const Vector3dVector& scan, const TimestampVector& timestamps) {
  // TODO: redundant max compute as its available after process_timestamps
  const auto max = std::max_element(timestamps.cbegin(), timestamps.cend());
  const Secondsd current_lidar_time = *max;

  if (lidar_state.time < EPSILON_TIME) {
    lidar_state.time = current_lidar_time;
    const auto& preproc_result = preprocess_scan(scan, config);
    if (!config.initialization_phase) {
      // use the first frame for the map only if we're not initializing
      map.Update(preproc_result.map_update_frame(), lidar_state.pose);
      poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);
      std::cout << "[INFO] Odometry map frame initialized with first lidar scan.\n";
    }
    return preproc_result.filtered_frame;
  }

  if (std::chrono::abs(current_lidar_time - lidar_state.time).count() > 1.0) {
    const double diff_seconds = (current_lidar_time - lidar_state.time).count();
    // TODO: std::expected with tl::expected (because ros humble)
    throw std::invalid_argument("Received LiDAR scan with " + std::to_string(diff_seconds) +
                                " seconds delta to previous scan.");
  }

  const auto& [avg_body_accel, avg_ang_vel] = std::invoke([&]() -> std::pair<Eigen::Vector3d, Eigen::Vector3d> {
    if (config.initialization_phase && !_initialized) {
      // assume static and
      initialize(current_lidar_time);
      return {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    }
    if (interval_stats.imu_count == 0) {
      std::cerr << "[WARNING] No Imu measurements in interval to average. Assuming constant velocity motion.\n";
      return {Eigen::Vector3d::Zero(), lidar_state.angular_velocity};
    }
    const Eigen::Vector3d avg_body_accel = interval_stats.body_acceleration_sum / interval_stats.imu_count;
    const Eigen::Vector3d avg_ang_vel = interval_stats.angular_velocity_sum / interval_stats.imu_count;
    if (avg_body_accel.norm() > 50.0) {
      std::cerr << "[WARNING] Erratic body acceleration computed, norm > 50 m/s2. Either IMU data is corrupted, or you "
                   "should report an issue.";
    }
    return {avg_body_accel, avg_ang_vel};
  });

  // compute relative motion using controls
  auto relative_pose_at_time = [&](const Secondsd time) -> Sophus::SE3d {
    const double dt = (time - lidar_state.time).count();
    Eigen::Matrix<double, 6, 1> tau;
    tau.head<3>() = lidar_state.velocity * dt + (avg_body_accel * square(dt) / 2);
    tau.tail<3>() = avg_ang_vel * dt;
    return Sophus::SE3d::exp(tau);
  };

  const Sophus::SE3d initial_guess = lidar_state.pose * relative_pose_at_time(current_lidar_time);

  // body acceleration filter
  const auto& accel_filter_info = get_accel_info(initial_guess.so3(), current_lidar_time);

  const auto& preproc_result = preprocess_scan(scan, timestamps, current_lidar_time, relative_pose_at_time, config);

  if (preproc_result.keypoints.size() < 10) {
    const std::string error_msg =
        "Keypoints for ICP registration = " + std::to_string(preproc_result.keypoints.size()) +
        ", this is too little for ICP and likely unintended. Input scan size = " + std::to_string(scan.size()) +
        ". Config voxel size = " + std::to_string(config.voxel_size) +
        ". Either the input scan is corrupt (empty) or the downsampling is too aggressive.";
    throw std::invalid_argument(error_msg);
  }

  // Optional dynamic-segmentation weighting context for ICP. Reading from
  // voxel_dyn during the parallel reduce is safe because we only write to it
  // after the loop exits, sequentially in this same thread.
  const DynWeightingContext dyn_ctx_storage = {.voxel_dyn = &voxel_dyn,
                                                .weight_decay_k = config.dyn_weight_decay_k,
                                                .min_hits_to_trust = config.dyn_min_hits_to_trust};
  const DynWeightingContext* dyn_ctx_ptr = config.dynamic_segmentation_enabled ? &dyn_ctx_storage : nullptr;

  Sophus::SE3d optimized_pose = initial_guess;
  bool ran_icp = false;
  if (!map.Empty()) {
    SCOPED_PROFILER("ICP");
    // Primary ICP runs without the camera residual so the `odom -> base_link`
    // path stays purely LIO-driven and continues to satisfy REP-105's "odom
    // is smooth, never jumps" contract. The camera-aided pose is computed
    // separately below at keyframes for the `map -> odom` correction.
    optimized_pose = icp(preproc_result.keypoints, map, initial_guess, config, accel_filter_info, dyn_ctx_ptr,
                          /*camera_ctx=*/nullptr);
    ran_icp = true;

    // estimate velocities and accelerations from the new pose
    const double dt = (current_lidar_time - lidar_state.time).count();
    const Sophus::SE3d motion = lidar_state.pose.inverse() * optimized_pose;
    const Eigen::Vector6d local_velocity = motion.log() / dt;
    const Eigen::Vector3d local_linear_acceleration =
        (local_velocity.head<3>() - motion.so3().inverse() * lidar_state.velocity) / dt;

    // update
    lidar_state.pose = optimized_pose;
    lidar_state.velocity = local_velocity.head<3>();
    lidar_state.angular_velocity = local_velocity.tail<3>();
    lidar_state.linear_acceleration = local_linear_acceleration;

    _imu_local_rotation = optimized_pose.so3(); // correct the drift in imu integration
  }
  // even if map is empty, time should still update
  lidar_state.time = current_lidar_time;
  _imu_local_rotation_time = current_lidar_time;

  // reset imu averages
  interval_stats.reset();

  // Camera keyframe path (Option A): once the gates pass, re-run ICP starting
  // from the LIO-only optimum with the camera residual switched in to get
  // T_combined. The delta against T_lio_only is broadcast as `map -> odom`.
  // The camera frame is consumed at most once per call.
  const bool camera_gate = config.camera_enabled && _camera_extrinsic_set && _pending_camera_frame.has_value() &&
                            ran_icp && static_cast<int>(_scan_counter) >= config.camera_warmup_scans &&
                            _pending_camera_frame->time <= current_lidar_time;
  bool keyframe_motion_ok = !_last_camera_keyframe_valid;
  bool keyframe_time_ok = !_last_camera_keyframe_valid;
  if (camera_gate && _last_camera_keyframe_valid) {
    const Sophus::SE3d motion_since_kf = _last_camera_keyframe_pose.inverse() * optimized_pose;
    const double translation_norm = motion_since_kf.translation().norm();
    const double rotation_angle = motion_since_kf.so3().log().norm();
    keyframe_motion_ok = translation_norm >= config.camera_keyframe_motion_threshold_m ||
                         rotation_angle >= config.camera_keyframe_rotation_threshold_rad;
    keyframe_time_ok =
        (current_lidar_time - _last_camera_keyframe_time).count() >= config.camera_keyframe_min_dt_s;
  }
  if (camera_gate && keyframe_motion_ok && keyframe_time_ok) {
    SCOPED_PROFILER("CAMERA_KEYFRAME");
    const auto* voxel_dyn_for_camera = config.dynamic_segmentation_enabled ? &voxel_dyn : nullptr;
    CameraVisibleSet camera_visible = compute_camera_visible_set(
        map, optimized_pose, _extrinsic_cam2base, *_pending_camera_frame, config, voxel_dyn_for_camera);
    if (static_cast<int>(camera_visible.points_odom.size()) >= config.camera_min_visible_points) {
      CameraResidualContext camera_ctx{};
      camera_ctx.frame = &(*_pending_camera_frame);
      camera_ctx.extrinsic_cam_to_base = &_extrinsic_cam2base;
      camera_ctx.visible = &camera_visible;
      const Sophus::SE3d camera_optimized_pose =
          icp(preproc_result.keypoints, map, optimized_pose, config, accel_filter_info, dyn_ctx_ptr, &camera_ctx);
      _map_to_odom = camera_optimized_pose * optimized_pose.inverse();
      _last_camera_keyframe_pose = optimized_pose;
      _last_camera_keyframe_time = current_lidar_time;
      _last_camera_keyframe_valid = true;
    }
  }
  // Consume the pending frame unconditionally so we never reuse a stale image:
  // either it was applied above, or its motion/time/visibility gates failed and
  // we move on to the next image instead.
  if (_pending_camera_frame.has_value() && _pending_camera_frame->time <= current_lidar_time) {
    _pending_camera_frame.reset();
  }

  // Dynamic-segmentation post-pass: learn from the converged ICP residuals
  // over the keypoints, then mask the map update with the freshly updated
  // dyn_score so we never add dynamic returns to the map.
  ++_scan_counter;
  const auto& map_update_frame = preproc_result.map_update_frame();
  if (config.dynamic_segmentation_enabled && ran_icp && !map.Empty()) {
    SCOPED_PROFILER("DYN_SEGMENTATION");
    const auto telemetry =
        collect_icp_telemetry(optimized_pose, preproc_result.keypoints, map, config.max_correspondance_distance);
    update_voxel_dyn_stats(voxel_dyn, telemetry, config, _scan_counter);
    const auto accept_mask = build_map_accept_mask(map_update_frame, lidar_state.pose, map, voxel_dyn, config);
    map.Update(map_update_frame, accept_mask, lidar_state.pose);
  } else {
    map.Update(map_update_frame, lidar_state.pose);
  }

  poses_with_timestamps.emplace_back(lidar_state.time, lidar_state.pose);

  return preproc_result.filtered_frame;
}

Vector3dVector LIO::register_scan(const Sophus::SE3d& extrinsic_lidar2base,
                                  const Vector3dVector& scan,
                                  const TimestampVector& timestamps) {
  if (extrinsic_lidar2base.log().norm() < EPSILON) {
    return register_scan(scan, timestamps);
  }

  Vector3dVector transformed_scan = scan;
  transform_points(extrinsic_lidar2base, transformed_scan);
  Vector3dVector frame = register_scan(transformed_scan, timestamps);
  transform_points(extrinsic_lidar2base.inverse(), frame);
  return frame;
}

// ============================ camera ===============================

void LIO::set_camera_extrinsic(const Sophus::SE3d& extrinsic_cam2base) {
  _extrinsic_cam2base = extrinsic_cam2base;
  _camera_extrinsic_set = true;
}

void LIO::add_camera_frame(const CameraFrame& frame) {
  // Replace the pending frame; the registration loop picks up the most recent
  // image whose timestamp is older than the current scan's max timestamp.
  _pending_camera_frame = frame;
}
} // namespace rko_lio::core
