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

/**
 * @file lio.hpp
 * Core LIO class and utilities for RKO-LIO.
 */

#pragma once
#include "sparse_voxel_grid.hpp"
#include "util.hpp"
#include <bonxai/grid_coord.hpp>
#include <optional>
#include <unordered_map>
#include <vector>

/** Core namespace containing LIO data structures and state definitions. */
namespace rko_lio::core {

/**
 * Hash functor for `Bonxai::CoordT`, used to key the `LIO::voxel_dyn` side
 * table. Bonxai itself provides a `std::hash<Bonxai::CoordT>` specialization
 * (see `bonxai/grid_coord.hpp`); aliasing it here lets the call sites be
 * explicit about which hash they depend on so the plan's contract is
 * preserved if upstream ever drops the specialization.
 */
using CoordTHash = std::hash<Bonxai::CoordT>;

/** Core LiDAR-inertial odometry algorithm class. */
class LIO {
public:
  /** Configuration parameters for odometry. */
  struct Config {
    /** Enable scan deskewing. */
    bool deskew = true;

    /** Maximum number of ICP iterations. */
    size_t max_iterations = 100;

    /** Size of voxel grid (m). */
    double voxel_size = 1.0;

    /** Max points per voxel. */
    int max_points_per_voxel = 20;

    /** Maximum lidar range (m). */
    double max_range = 100.0;

    /** Minimum lidar range (m). */
    double min_range = 1.0;

    /** ICP convergence threshold. */
    double convergence_criterion = 1e-5;

    /** Max distance for correspondences (m). */
    double max_correspondance_distance = 0.5;

    /** Thread count for data association (0 = automatic). */
    int max_num_threads = 0;

    /** Enable initialization phase. */
    bool initialization_phase = false;

    /** Maximum expected jerk (m/s³). */
    double max_expected_jerk = 3;

    /** Enable double downsampling. */
    bool double_downsample = true;

    /** Minimum weight for orientation regularization. */
    double min_beta = 200;

    // ---- Camera tight-coupling (opt-in) ----

    /** Enable the camera edge-alignment residual block inside ICP. */
    bool camera_enabled = false;

    /** Scalar weight applied to the camera linear system before summing into the GN system. */
    double camera_weight = 1.0;

    /** Huber clip on the distance-transform residual, in pixels. */
    double camera_max_dt_residual_px = 20.0;

    /** Below this count of visible projected points the camera block is skipped for the scan. */
    int camera_min_visible_points = 200;

    /** Map points closer than this distance from the camera (in m) are culled. */
    double camera_min_point_depth_m = 0.5;

    /** Optional low-resolution z-buffer occlusion test during the visible-point pre-pass. */
    bool camera_use_occlusion_zbuf = true;

    /** Z-buffer downsample factor (power-of-2 recommended) for the occlusion test. */
    int camera_zbuf_downsample = 4;

    /** Skip the camera block for the first N scans after startup so the map can populate. */
    int camera_warmup_scans = 5;

    // ---- Dynamic-point segmentation (opt-in) ----

    /** Enable per-voxel dynamic statistics maintained from ICP correspondence residuals. */
    bool dynamic_segmentation_enabled = false;

    /** Residual (m) below which a correspondence is considered definitely static. */
    double dyn_tau_static_m = 0.10;

    /** Residual (m) above which a correspondence is treated as dynamic-suspicious. */
    double dyn_tau_dynamic_m = 0.30;

    /** EMA mixing factor for resid_ema and dyn_score. */
    double dyn_ema_alpha = 0.20;

    /** Voxels with `dyn_score` above this gate are skipped from `SparseVoxelGrid::Update`. */
    double dyn_skip_map_score = 0.60;

    /** Weight decay factor: w_i = exp(-k * dyn_score) for ICP weighting on the next scan. */
    double dyn_weight_decay_k = 4.0;

    /** A voxel needs at least this many hits before its `dyn_score` is trusted. */
    int dyn_min_hits_to_trust = 3;
  };

  /** Configuration parameters. */
  Config config;

  /** Local map as sparse voxel grid (Bonxai). */
  SparseVoxelGrid map;

  /** Current LiDAR state estimate. */
  State lidar_state;

  /** IMU bias estimates when initialization is enabled. */
  ImuBias imu_bias;

  /** Mean body acceleration estimate. */
  Eigen::Vector3d mean_body_acceleration = Eigen::Vector3d::Zero();

  /** Covariance of body acceleration estimate. */
  Eigen::Matrix3d body_acceleration_covariance = Eigen::Matrix3d::Identity();

  /** IMU measurement statistics since last LiDAR frame. */
  IntervalStats interval_stats;

  explicit LIO(const Config& config_)
      : config(config_), map(config_.voxel_size, config_.max_range, config_.max_points_per_voxel) {}

  /** Add an IMU measurement expressed in the base frame. */
  void add_imu_measurement(const ImuControl& base_imu);

  /**
   * Add an IMU measurement expressed in the IMU frame and transform it
   * to the base frame using the given extrinsic calibration.
   * @param extrinsic_imu2base Extrinsic transform from IMU to base frame.
   * @param raw_imu Raw IMU measurement.
   */
  void add_imu_measurement(const Sophus::SE3d& extrinsic_imu2base, const ImuControl& raw_imu);

  /**
   * Register a LiDAR scan, applying deskewing based on the initial motion guess
   * and clipping points beyond valid range.
   * @param scan Input raw point cloud.
   * @param timestamps Absolute timestamps corresponding to each scan point.
   * @return Deskewed and clipped point cloud.
   */
  Vector3dVector register_scan(const Vector3dVector& scan, const TimestampVector& timestamps);

  /**
   * Register a LiDAR scan for which the extrinsic calibration from lidar to base
   * has already been applied.
   * @param extrinsic_lidar2base Extrinsic from lidar to base frame.
   * @param scan Input raw point cloud.
   * @param timestamps Absolute timestamps corresponding to each scan point.
   * @return Deskewed and clipped scan in the original lidar frame.
   */
  Vector3dVector register_scan(const Sophus::SE3d& extrinsic_lidar2base,
                               const Vector3dVector& scan,
                               const TimestampVector& timestamps);

  /** Sequence of registered scan poses with corresponding timestamps. */
  std::vector<std::pair<Secondsd, Sophus::SE3d>> poses_with_timestamps;

  /**
   * Register a fixed extrinsic from the camera frame to the base frame.
   * Required before `add_camera_frame` will produce useful corrections.
   */
  void set_camera_extrinsic(const Sophus::SE3d& extrinsic_cam2base);

  /**
   * Submit the most recent rectified camera frame (with precomputed
   * distance-transform image) for tight coupling into the next scan's ICP.
   *
   * Calling this overwrites any previously-submitted but still-unconsumed
   * frame; the registration loop only ever uses the latest frame whose
   * timestamp is older than the current scan's max timestamp.
   */
  void add_camera_frame(const CameraFrame& frame);

  /**
   * The `map -> odom` correction maintained from camera keyframes (Option A
   * in the design doc). Identity when no camera keyframe has produced a
   * correction yet. Always safe to read; updates are atomic from the
   * outside-caller's point of view.
   */
  const Sophus::SE3d& map_to_odom() const { return _map_to_odom; }

  /** Whether `set_camera_extrinsic` has been called with a non-trivial transform. */
  bool camera_extrinsic_set() const { return _camera_extrinsic_set; }

  /**
   * Per-voxel dynamic statistics side-table, parallel to `map`. Empty unless
   * `config.dynamic_segmentation_enabled = true`. Exposed for visualisation
   * and downstream consumers.
   */
  std::unordered_map<Bonxai::CoordT, VoxelDynStats, CoordTHash> voxel_dyn;

private:
  /**
   * Initialize internal odometry state using the given lidar timestamp.
   * @param lidar_time Current lidar timestamp.
   */
  void initialize(const Secondsd lidar_time);

  /** get the convenience struct with accel mag variance and local gravity estimate. */
  std::optional<AccelInfo> get_accel_info(const Sophus::SO3d& rotation_estimate, const Secondsd& time);

  /** True if odometry initialization has been completed. */
  bool _initialized = false;

  /** Latest IMU orientation used for gravity compensation. This is ahead of the rotation in the state. */
  Sophus::SO3d _imu_local_rotation;

  /** Timestamp of the latest IMU orientation. Once a scan is registered, this is reset to the lidar state orientation.
   */
  Secondsd _imu_local_rotation_time = Secondsd{0.0};

  /** Timestamp of the most recent real IMU measurement. */
  Secondsd _last_real_imu_time = Secondsd{0.0};

  /** Angular velocity of last true IMU measurement expressed in base frame. */
  Eigen::Vector3d _last_real_base_imu_ang_vel = Eigen::Vector3d::Zero();

  /** Fixed extrinsic from camera frame to base frame. */
  Sophus::SE3d _extrinsic_cam2base;
  bool _camera_extrinsic_set = false;

  /** Latest camera frame submitted via `add_camera_frame`, consumed at most once. */
  std::optional<CameraFrame> _pending_camera_frame;

  /** Monotonic scan counter, used for the camera warmup gate and `last_seen_scan`. */
  uint32_t _scan_counter = 0;

  /** map -> odom rigid correction (REP-105). Identity until a camera keyframe lands. */
  Sophus::SE3d _map_to_odom;
};
} // namespace rko_lio::core
