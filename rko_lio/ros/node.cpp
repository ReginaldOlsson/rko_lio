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

#include "node.hpp"
#include "rko_lio/core/process_timestamps.hpp"
#include "rko_lio/core/profiler.hpp"
#include "rko_lio/ros/utils/utils.hpp"
// other
#include <cv_bridge/cv_bridge.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/serialization.hpp>
#include <stdexcept>

namespace {
using namespace std::literals;

rko_lio::core::ImuControl imu_msg_to_imu_data(const sensor_msgs::msg::Imu& imu_msg) {
  rko_lio::core::ImuControl imu_data;
  imu_data.time = rko_lio::ros::utils::ros_time_to_seconds(imu_msg.header.stamp);
  imu_data.angular_velocity = rko_lio::ros::utils::ros_xyz_to_eigen_vector3d(imu_msg.angular_velocity);
  imu_data.acceleration = rko_lio::ros::utils::ros_xyz_to_eigen_vector3d(imu_msg.linear_acceleration);
  return imu_data;
}

} // namespace

namespace rko_lio::core {
// necessary for serializing the config, including the namespacing
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LIO::Config,
                                   deskew,
                                   max_iterations,
                                   voxel_size,
                                   max_points_per_voxel,
                                   max_range,
                                   min_range,
                                   convergence_criterion,
                                   max_correspondance_distance,
                                   max_num_threads,
                                   initialization_phase,
                                   max_expected_jerk,
                                   double_downsample,
                                   min_beta,
                                   camera_enabled,
                                   camera_weight,
                                   camera_max_dt_residual_px,
                                   camera_min_visible_points,
                                   camera_min_point_depth_m,
                                   camera_use_occlusion_zbuf,
                                   camera_zbuf_downsample,
                                   camera_warmup_scans,
                                   camera_keyframe_motion_threshold_m,
                                   camera_keyframe_rotation_threshold_rad,
                                   camera_keyframe_min_dt_s,
                                   dynamic_segmentation_enabled,
                                   dyn_tau_static_m,
                                   dyn_tau_dynamic_m,
                                   dyn_ema_alpha,
                                   dyn_skip_map_score,
                                   dyn_weight_decay_k,
                                   dyn_min_hits_to_trust)
} // namespace rko_lio::core

namespace rko_lio::ros {

Node::Node(const std::string& node_name, const rclcpp::NodeOptions& options) {
  node = rclcpp::Node::make_shared(node_name, options);
  imu_topic = node->declare_parameter<std::string>("imu_topic");     // required
  lidar_topic = node->declare_parameter<std::string>("lidar_topic"); // required
  base_frame = node->declare_parameter<std::string>("base_frame");   // required
  imu_frame = node->declare_parameter<std::string>("imu_frame", imu_frame);
  lidar_frame = node->declare_parameter<std::string>("lidar_frame", lidar_frame);
  odom_frame = node->declare_parameter<std::string>("odom_frame", odom_frame);
  odom_topic = node->declare_parameter<std::string>("odom_topic", odom_topic);

  // tf
  invert_odom_tf = node->declare_parameter<bool>("invert_odom_tf", invert_odom_tf);
  tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
  tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*node);

  // publishing
  const rclcpp::QoS publisher_qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
  odom_publisher = node->create_publisher<nav_msgs::msg::Odometry>(odom_topic, publisher_qos);

  publish_lidar_acceleration = node->declare_parameter<bool>("publish_lidar_acceleration", publish_lidar_acceleration);
  if (publish_lidar_acceleration) {
    lidar_accel_publisher =
        node->create_publisher<geometry_msgs::msg::AccelStamped>("rko_lio/lidar_acceleration", publisher_qos);
  }

  publish_deskewed_scan = node->declare_parameter<bool>("publish_deskewed_scan", publish_deskewed_scan);
  if (publish_deskewed_scan) {
    deskewed_scan_topic = node->declare_parameter<std::string>("deskewed_scan_topic", deskewed_scan_topic);
    frame_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(deskewed_scan_topic, publisher_qos);
  }

  publish_local_map = node->declare_parameter<bool>("publish_local_map", publish_local_map);
  if (publish_local_map) {
    map_topic = node->declare_parameter<std::string>("map_topic", map_topic);
    publish_map_after = core::Secondsd(node->declare_parameter<double>("publish_map_after", publish_map_after.count()));
    map_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(map_topic, publisher_qos);
    map_publish_thead = std::jthread([this]() { publish_map_loop(); });
  }

  // lio params
  core::LIO::Config lio_config{};
  lio_config.deskew = node->declare_parameter<bool>("deskew", lio_config.deskew);
  lio_config.max_iterations =
      static_cast<size_t>(node->declare_parameter<int>("max_iterations", static_cast<int>(lio_config.max_iterations)));
  lio_config.voxel_size = node->declare_parameter<double>("voxel_size", lio_config.voxel_size);
  lio_config.max_points_per_voxel =
      static_cast<int>(node->declare_parameter<int>("max_points_per_voxel", lio_config.max_points_per_voxel));
  lio_config.max_range = node->declare_parameter<double>("max_range", lio_config.max_range);
  lio_config.min_range = node->declare_parameter<double>("min_range", lio_config.min_range);
  lio_config.convergence_criterion =
      node->declare_parameter<double>("convergence_criterion", lio_config.convergence_criterion);
  lio_config.max_correspondance_distance =
      node->declare_parameter<double>("max_correspondance_distance", lio_config.max_correspondance_distance);
  lio_config.max_num_threads =
      static_cast<int>(node->declare_parameter<int>("max_num_threads", lio_config.max_num_threads));
  lio_config.initialization_phase =
      node->declare_parameter<bool>("initialization_phase", lio_config.initialization_phase);
  lio_config.max_expected_jerk = node->declare_parameter<double>("max_expected_jerk", lio_config.max_expected_jerk);
  lio_config.double_downsample = node->declare_parameter<bool>("double_downsample", lio_config.double_downsample);
  lio_config.min_beta = node->declare_parameter<double>("min_beta", lio_config.min_beta);

  // ---- camera tight-coupling parameters (opt-in via camera_enabled) ----
  lio_config.camera_enabled = node->declare_parameter<bool>("camera_enabled", lio_config.camera_enabled);
  lio_config.camera_weight = node->declare_parameter<double>("camera_weight", lio_config.camera_weight);
  lio_config.camera_max_dt_residual_px =
      node->declare_parameter<double>("camera_max_dt_residual_px", lio_config.camera_max_dt_residual_px);
  lio_config.camera_min_visible_points =
      node->declare_parameter<int>("camera_min_visible_points", lio_config.camera_min_visible_points);
  lio_config.camera_min_point_depth_m =
      node->declare_parameter<double>("camera_min_point_depth_m", lio_config.camera_min_point_depth_m);
  lio_config.camera_use_occlusion_zbuf =
      node->declare_parameter<bool>("camera_use_occlusion_zbuf", lio_config.camera_use_occlusion_zbuf);
  lio_config.camera_zbuf_downsample =
      node->declare_parameter<int>("camera_zbuf_downsample", lio_config.camera_zbuf_downsample);
  lio_config.camera_warmup_scans = node->declare_parameter<int>("camera_warmup_scans", lio_config.camera_warmup_scans);
  lio_config.camera_keyframe_motion_threshold_m = node->declare_parameter<double>(
      "camera_keyframe_motion_threshold_m", lio_config.camera_keyframe_motion_threshold_m);
  lio_config.camera_keyframe_rotation_threshold_rad = node->declare_parameter<double>(
      "camera_keyframe_rotation_threshold_rad", lio_config.camera_keyframe_rotation_threshold_rad);
  lio_config.camera_keyframe_min_dt_s =
      node->declare_parameter<double>("camera_keyframe_min_dt_s", lio_config.camera_keyframe_min_dt_s);

  // ---- dynamic-segmentation parameters (opt-in via dynamic_segmentation_enabled) ----
  lio_config.dynamic_segmentation_enabled =
      node->declare_parameter<bool>("dynamic_segmentation_enabled", lio_config.dynamic_segmentation_enabled);
  lio_config.dyn_tau_static_m = node->declare_parameter<double>("dyn_tau_static_m", lio_config.dyn_tau_static_m);
  lio_config.dyn_tau_dynamic_m = node->declare_parameter<double>("dyn_tau_dynamic_m", lio_config.dyn_tau_dynamic_m);
  lio_config.dyn_ema_alpha = node->declare_parameter<double>("dyn_ema_alpha", lio_config.dyn_ema_alpha);
  lio_config.dyn_skip_map_score = node->declare_parameter<double>("dyn_skip_map_score", lio_config.dyn_skip_map_score);
  lio_config.dyn_weight_decay_k = node->declare_parameter<double>("dyn_weight_decay_k", lio_config.dyn_weight_decay_k);
  lio_config.dyn_min_hits_to_trust =
      node->declare_parameter<int>("dyn_min_hits_to_trust", lio_config.dyn_min_hits_to_trust);

  lio = std::make_unique<core::LIO>(lio_config);

  // ROS-side camera parameters (topics, frames, Canny knobs).
  image_topic = node->declare_parameter<std::string>("image_topic", image_topic);
  camera_info_topic = node->declare_parameter<std::string>("camera_info_topic", camera_info_topic);
  camera_frame = node->declare_parameter<std::string>("camera_frame", camera_frame);
  canny_low_threshold = node->declare_parameter<double>("canny_low_threshold", canny_low_threshold);
  canny_high_threshold = node->declare_parameter<double>("canny_high_threshold", canny_high_threshold);
  canny_aperture_size = node->declare_parameter<int>("canny_aperture_size", canny_aperture_size);
  image_max_rate_hz = node->declare_parameter<double>("image_max_rate_hz", image_max_rate_hz);
  map_frame = node->declare_parameter<std::string>("map_frame", map_frame);
  publish_map_to_odom_tf = node->declare_parameter<bool>("publish_map_to_odom_tf", publish_map_to_odom_tf);

  // Publish dynamic-segmentation split topics if requested. Only meaningful
  // when both `publish_deskewed_scan` and `dynamic_segmentation_enabled` are
  // on; we silently no-op otherwise to keep the toggle independent of the
  // other features.
  publish_dynamic_split = node->declare_parameter<bool>("publish_dynamic_split", publish_dynamic_split) &&
                           lio_config.dynamic_segmentation_enabled && publish_deskewed_scan;
  if (publish_dynamic_split) {
    deskewed_scan_static_topic =
        node->declare_parameter<std::string>("deskewed_scan_static_topic", deskewed_scan_static_topic);
    deskewed_scan_dynamic_topic =
        node->declare_parameter<std::string>("deskewed_scan_dynamic_topic", deskewed_scan_dynamic_topic);
    frame_static_publisher =
        node->create_publisher<sensor_msgs::msg::PointCloud2>(deskewed_scan_static_topic, publisher_qos);
    frame_dynamic_publisher =
        node->create_publisher<sensor_msgs::msg::PointCloud2>(deskewed_scan_dynamic_topic, publisher_qos);
  }

  // Camera subscriptions only when the feature is enabled at startup.
  if (lio_config.camera_enabled) {
    camera_info_sub = node->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic, rclcpp::QoS(5),
        [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) { camera_info_callback(msg); });
    image_subscriber = image_transport::create_subscription(
        node.get(), image_topic,
        [this](const sensor_msgs::msg::Image::ConstSharedPtr& msg) { image_callback(msg); }, "raw",
        rclcpp::QoS(rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()).get_rmw_qos_profile());
    RCLCPP_INFO_STREAM(node->get_logger(), "Camera coupling enabled. Subscribing to image: " << image_topic
                                                                                              << " and camera info: "
                                                                                              << camera_info_topic);
  }

  // Timestamp processing params - lts for lidar time stamps, without having 100 char param names
  timestamp_proc_config.multiplier_to_seconds =
      node->declare_parameter<double>("lts_multiplier_to_seconds", timestamp_proc_config.multiplier_to_seconds);
  timestamp_proc_config.force_absolute =
      node->declare_parameter<bool>("lts_force_absolute", timestamp_proc_config.force_absolute);
  timestamp_proc_config.force_relative =
      node->declare_parameter<bool>("lts_force_relative", timestamp_proc_config.force_relative);

  // manually, if, define extrinsics
  parse_cli_extrinsics();

  RCLCPP_INFO_STREAM(node->get_logger(),
                     "Subscribed to IMU: "
                         << imu_topic << (!imu_frame.empty() ? " (frame " + imu_frame + ")" : "") << " and LiDAR: "
                         << lidar_topic << (!lidar_frame.empty() ? " (frame " + lidar_frame + ")" : "")
                         << ". Max number of threads: " << lio_config.max_num_threads << ". Publishing odometry to "
                         << odom_topic << " ( " << odom_frame
                         << " ) and acceleration "
                            "estimates to rko_lio/lidar_acceleration. Deskewing is "
                         << (lio->config.deskew ? "enabled" : "disabled") << "."
                         << (publish_deskewed_scan ? (" Publishing deskewed_cloud to " + deskewed_scan_topic + ".")
                                                   : ""));

  // disk logging
  dump_results = node->declare_parameter<bool>("dump_results", dump_results);
  results_dir = node->declare_parameter<std::string>("results_dir", results_dir);
  run_name = node->declare_parameter<std::string>("run_name", run_name);
  rclcpp::on_shutdown([this]() {
    // i'll need to look into rclcpp::Context a bit more, but for now i think this callback should be called before
    // anything gets destroyed.
    if (dump_results) {
      // it is probably still a veery good idea to make dump_results_to_disk noexcept
      dump_results_to_disk(results_dir, run_name);
    }
  });

  registration_thread = std::jthread([this]() { registration_loop(); });

  RCLCPP_INFO(node->get_logger(), "RKO LIO Node is up!");
}

void Node::parse_cli_extrinsics() {
  auto parse_extrinsic = [this](const std::string& name, Sophus::SE3d& extrinsic) {
    const std::string param_name = "extrinsic_" + name + "2base_quat_xyzw_xyz";
    const std::vector<double> vec = node->declare_parameter<std::vector<double>>(param_name, std::vector<double>{});

    if (vec.size() != 7) {
      if (!vec.empty()) {
        RCLCPP_WARN_STREAM(node->get_logger(),
                           "Parameter 'extrinsic_"
                               << name << "2base_quat_xyzw_xyz' is set but has wrong size: " << vec.size()
                               << ". Expected 7 (qx, qy, qz, qw, x, y, z). check the value: "
                               << Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size()).transpose());
      }
      return false;
    }
    Eigen::Quaterniond q(vec[3], vec[0], vec[1], vec[2]); // qw, qx, qy, qz
    if (q.norm() < 1e-6) {
      throw std::runtime_error(name + " extrinsic quaternion has zero norm");
    }
    extrinsic = Sophus::SE3d(q, Eigen::Vector3d(vec[4], vec[5], vec[6]));
    RCLCPP_INFO_STREAM(node->get_logger(), "Parsed " << name << " extrinsic as: " << extrinsic.log().transpose());
    return true;
  };
  const bool imu_ok = parse_extrinsic("imu", extrinsic_imu2base);
  const bool lidar_ok = parse_extrinsic("lidar", extrinsic_lidar2base);
  extrinsics_set = imu_ok && lidar_ok;
}

bool Node::check_and_set_extrinsics() {
  // Camera resolution is best-effort and decoupled from the imu/lidar gate
  // so the rest of the pipeline can start before the camera tf is published.
  try_set_camera_extrinsic();

  if (extrinsics_set) {
    return true;
  }
  const std::optional<Sophus::SE3d> imu_transform = utils::get_transform(tf_buffer, imu_frame, base_frame, 0s);
  if (!imu_transform) {
    return false;
  }
  const std::optional<Sophus::SE3d> lidar_transform = utils::get_transform(tf_buffer, lidar_frame, base_frame, 0s);
  if (!lidar_transform) {
    return false;
  }
  extrinsic_imu2base = imu_transform.value();
  extrinsic_lidar2base = lidar_transform.value();
  extrinsics_set = true;
  return true;
}

void Node::try_set_camera_extrinsic() {
  if (!lio || !lio->config.camera_enabled || lio->camera_extrinsic_set() || camera_frame.empty() ||
      base_frame.empty()) {
    return;
  }
  const std::optional<Sophus::SE3d> cam = utils::get_transform(tf_buffer, camera_frame, base_frame, 0s);
  if (cam) {
    lio->set_camera_extrinsic(*cam);
    RCLCPP_INFO_STREAM(node->get_logger(),
                       "Camera extrinsic cam2base resolved as: " << cam->log().transpose());
  }
}

void Node::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg) {
  if (imu_frame.empty()) {
    if (imu_msg->header.frame_id.empty() && !extrinsics_set) {
      throw std::runtime_error("IMU message header has no frame id and we need it to query TF for the extrinsics. "
                               "Either specify the frame id or the extrinsic manually.");
    }
    imu_frame = imu_msg->header.frame_id;
    RCLCPP_INFO_STREAM(node->get_logger(), "Parsed the imu frame id as: " << imu_frame);
  }
  if (!check_and_set_extrinsics()) {
    // we assume that extrinsics are static. if they change, its better to query the tf directly in the registration
    // loop for each message being processed asynchronously.
    return;
  }
  {
    std::lock_guard lock(buffer_mutex);
    imu_buffer.emplace(imu_msg_to_imu_data(*imu_msg));
    atomic_can_process = !lidar_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().timestamps.max;
  }
  if (atomic_can_process) {
    sync_condition_variable.notify_one();
  }
}

void Node::lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) {
  if (lidar_frame.empty()) {
    if (lidar_msg->header.frame_id.empty() && !extrinsics_set) {
      throw std::runtime_error("LiDAR message header has no frame id and we need it to query TF for the extrinsics. "
                               "Either specify the frame id or the extrinsic manually.");
    }
    lidar_frame = lidar_msg->header.frame_id;
    RCLCPP_INFO_STREAM(node->get_logger(), "Parsed the lidar frame id as: " << lidar_frame);
  }
  if (!check_and_set_extrinsics()) {
    return;
  }
  {
    std::lock_guard lock(buffer_mutex);
    if (lidar_buffer.size() >= max_lidar_buffer_size) {
      RCLCPP_WARN_STREAM(node->get_logger(), "Registration lidar buffer limit reached. Dropping frame.");
      sync_condition_variable.notify_one();
      return;
    }
  }
  try {
    const auto& [timestamps, scan] = std::invoke([&]() -> std::tuple<core::Timestamps, core::Vector3dVector> {
      const core::Secondsd& header_stamp = utils::ros_time_to_seconds(lidar_msg->header.stamp);
      if (lio->config.deskew) {
        const auto& [scan, raw_timestamps] = utils::point_cloud2_to_eigen_with_timestamps(lidar_msg);
        const core::Timestamps& timestamps =
            core::process_timestamps(raw_timestamps, header_stamp, timestamp_proc_config);
        return {timestamps, scan};
      } else {
        RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                                "Deskewing is disabled. Populating timestamps with static header time.");
        const core::Vector3dVector scan = utils::point_cloud2_to_eigen(lidar_msg);
        return {{.min = header_stamp, .max = header_stamp, .times = core::TimestampVector(scan.size(), header_stamp)},
                scan};
      }
    });

    {
      std::lock_guard lock(buffer_mutex);
      lidar_buffer.emplace(timestamps, scan);
      atomic_can_process = !imu_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().timestamps.max;
    }
    if (atomic_can_process) {
      sync_condition_variable.notify_one();
    }
  } catch (const std::invalid_argument& ex) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Encountered error, dropping frame: Error. " << ex.what());
  }
}

void Node::camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg) {
  if (camera_frame.empty()) {
    if (info_msg->header.frame_id.empty()) {
      RCLCPP_WARN_ONCE(node->get_logger(),
                       "CameraInfo header has no frame id; please set camera_frame or fix the producer.");
      return;
    }
    camera_frame = info_msg->header.frame_id;
    RCLCPP_INFO_STREAM(node->get_logger(), "Parsed the camera frame id as: " << camera_frame);
    try_set_camera_extrinsic();
  }

  std::lock_guard lock(camera_intrinsics_mutex);
  // Build P-matrix-based pinhole intrinsics. P[0:3,0:3] is the rectified
  // projection so fx, fy, cx, cy come from there. If P is empty, fall back
  // to K (assumes already-rectified images).
  const std::array<double, 12>& P = info_msg->p;
  const std::array<double, 9>& K = info_msg->k;
  const bool P_valid = P[0] != 0.0;
  camera_intrinsics.fx = P_valid ? P[0] : K[0];
  camera_intrinsics.fy = P_valid ? P[5] : K[4];
  camera_intrinsics.cx = P_valid ? P[2] : K[2];
  camera_intrinsics.cy = P_valid ? P[6] : K[5];
  camera_intrinsics.width = static_cast<int>(info_msg->width);
  camera_intrinsics.height = static_cast<int>(info_msg->height);

  // Build rectification maps if the image is distorted. We use cv::remap with
  // precomputed maps so each image-callback invocation only pays for one
  // bilinear remap rather than an undistort solve.
  bool has_distortion = false;
  for (double d : info_msg->d) {
    if (std::abs(d) > 1e-12) {
      has_distortion = true;
      break;
    }
  }
  needs_rectification = has_distortion;
  if (needs_rectification) {
    cv::Mat K_mat(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) {
      K_mat.at<double>(i / 3, i % 3) = K[i];
    }
    cv::Mat D_mat(static_cast<int>(info_msg->d.size()), 1, CV_64F);
    for (size_t i = 0; i < info_msg->d.size(); ++i) {
      D_mat.at<double>(static_cast<int>(i), 0) = info_msg->d[i];
    }
    cv::Mat R_mat = cv::Mat::eye(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) {
      // R is the stereo rectifying rotation; identity when monocular.
      if (info_msg->r[i] != 0.0) {
        R_mat.at<double>(i / 3, i % 3) = info_msg->r[i];
      }
    }
    cv::Mat P_mat(3, 4, CV_64F);
    for (int i = 0; i < 12; ++i) {
      P_mat.at<double>(i / 4, i % 4) = P[i];
    }
    const cv::Size size(static_cast<int>(info_msg->width), static_cast<int>(info_msg->height));
    cv::initUndistortRectifyMap(K_mat, D_mat, R_mat, P_mat, size, CV_16SC2, rectify_map_x, rectify_map_y);
  }

  camera_intrinsics_ready = true;
}

void Node::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg) {
  if (!lio || !lio->config.camera_enabled) {
    return;
  }

  const core::Secondsd stamp = utils::ros_time_to_seconds(image_msg->header.stamp);
  // Throttle to image_max_rate_hz so a high-rate camera does not starve the
  // image-callback thread with redundant DT computations.
  if (image_max_rate_hz > 0.0) {
    const double min_dt = 1.0 / image_max_rate_hz;
    if (last_image_processed_time.count() > 0.0 && (stamp - last_image_processed_time).count() < min_dt) {
      return;
    }
  }

  cv::Mat gray;
  try {
    auto cv_ptr = cv_bridge::toCvShare(image_msg, "mono8");
    gray = cv_ptr->image; // shares storage with the message; copy on Canny below
  } catch (const cv_bridge::Exception& e) {
    RCLCPP_WARN_STREAM(node->get_logger(), "cv_bridge failure: " << e.what());
    return;
  }
  if (gray.empty()) {
    return;
  }

  cv::Mat rectified;
  core::PinholeIntrinsics intr;
  {
    std::lock_guard lock(camera_intrinsics_mutex);
    if (!camera_intrinsics_ready) {
      RCLCPP_WARN_ONCE(node->get_logger(), "Dropping camera image: CameraInfo not yet received.");
      return;
    }
    if (needs_rectification) {
      cv::remap(gray, rectified, rectify_map_x, rectify_map_y, cv::INTER_LINEAR);
    } else {
      rectified = gray.clone();
    }
    intr = camera_intrinsics;
  }
  if (rectified.empty()) {
    return;
  }
  // Make sure dimensions match the CameraInfo (some drivers stream different
  // sizes briefly during startup).
  if (rectified.cols != intr.width || rectified.rows != intr.height) {
    return;
  }

  cv::Mat edges;
  cv::Canny(rectified, edges, canny_low_threshold, canny_high_threshold, canny_aperture_size);

  // cv::distanceTransform requires an "inverted" mask: zeros at edge pixels,
  // non-zero elsewhere.
  cv::Mat inv;
  cv::bitwise_not(edges, inv);
  cv::Mat dt_image;
  cv::distanceTransform(inv, dt_image, cv::DIST_L2, 3);

  // Pack into a shared float buffer the core can keep alive across threads.
  auto buf = std::make_shared<std::vector<float>>(static_cast<size_t>(dt_image.rows) *
                                                  static_cast<size_t>(dt_image.cols));
  if (dt_image.isContinuous()) {
    std::memcpy(buf->data(), dt_image.ptr<float>(0), buf->size() * sizeof(float));
  } else {
    for (int r = 0; r < dt_image.rows; ++r) {
      std::memcpy(buf->data() + static_cast<size_t>(r) * static_cast<size_t>(dt_image.cols),
                  dt_image.ptr<float>(r), static_cast<size_t>(dt_image.cols) * sizeof(float));
    }
  }
  core::CameraFrame frame{};
  frame.time = stamp;
  frame.intrinsics = intr;
  frame.dt_image = buf;
  frame.rows = dt_image.rows;
  frame.cols = dt_image.cols;

  {
    std::lock_guard lock(buffer_mutex);
    if (camera_buffer.size() >= max_camera_buffer_size) {
      // Drop the oldest frame; we always prefer the most recent observation.
      camera_buffer.pop();
    }
    camera_buffer.push(std::move(frame));
  }

  last_image_processed_time = stamp;
}

void Node::drain_camera_buffer_for_scan(const core::Secondsd& scan_end_time) {
  if (!lio || !lio->config.camera_enabled) {
    return;
  }
  // Find the most recent camera frame with t_img <= t_scan_end. Drop everything older.
  std::optional<core::CameraFrame> latest;
  while (!camera_buffer.empty()) {
    if (camera_buffer.front().time > scan_end_time) {
      break; // remaining frames are in the future; leave them for the next scan
    }
    latest = std::move(camera_buffer.front());
    camera_buffer.pop();
  }
  if (latest) {
    lio->add_camera_frame(*latest);
  }
}

void Node::registration_loop() {
  while (rclcpp::ok() && atomic_node_running) {
    SCOPED_PROFILER("ROS Registration Loop");
    std::unique_lock buffer_lock(buffer_mutex);
    sync_condition_variable.wait(buffer_lock, [this]() { return !atomic_node_running || atomic_can_process; });
    if (!atomic_node_running) {
      // node could have been killed after waiting on the cv
      break;
    }
    core::LidarFrame frame = std::move(lidar_buffer.front());
    lidar_buffer.pop();
    const auto& [timestamps, scan] = frame;
    const auto& [start_stamp, end_stamp, time_vector] = timestamps;
    for (; !imu_buffer.empty() && imu_buffer.front().time < end_stamp; imu_buffer.pop()) {
      const core::ImuControl& imu_data = imu_buffer.front();
      lio->add_imu_measurement(extrinsic_imu2base, imu_data);
    }
    // Drain camera frames whose timestamps are no newer than this scan's end;
    // keep the most recent one (Option B in the design doc).
    drain_camera_buffer_for_scan(end_stamp);
    // check if there are more messages buffered already
    atomic_can_process =
        !imu_buffer.empty() && !lidar_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().timestamps.max;
    buffer_lock.unlock(); // we dont touch the buffers anymore

    try {
      const core::Vector3dVector deskewed_frame = std::invoke([&]() {
        if (publish_local_map) {
          std::lock_guard lock(local_map_mutex); // publish_map thread might access simultaneously
          return lio->register_scan(extrinsic_lidar2base, scan, time_vector);
        } else {
          return lio->register_scan(extrinsic_lidar2base, scan, time_vector);
        }
      });

      if (!deskewed_frame.empty()) {
        // TODO: first frame is skipped and an empty frame is returned. improve how we handle this
        if (publish_deskewed_scan) {
          std_msgs::msg::Header header;
          header.frame_id = lidar_frame;
          header.stamp = rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(end_stamp).count());
          frame_publisher->publish(utils::eigen_to_point_cloud2(deskewed_frame, header));
          if (publish_dynamic_split && lio->config.dynamic_segmentation_enabled) {
            const Sophus::SE3d lidar_to_odom = lio->lidar_state.pose * extrinsic_lidar2base;
            const auto& dyn_table = lio->voxel_dyn;
            const double gate = lio->config.dyn_skip_map_score;
            const int min_hits = lio->config.dyn_min_hits_to_trust;
            core::Vector3dVector static_points;
            core::Vector3dVector dynamic_points;
            static_points.reserve(deskewed_frame.size());
            dynamic_points.reserve(deskewed_frame.size() / 8);
            for (const auto& p : deskewed_frame) {
              const Eigen::Vector3d p_odom = lidar_to_odom * p;
              const Bonxai::CoordT key = lio->map.PosToCoord(p_odom);
              const auto it = dyn_table.find(key);
              const bool is_dynamic = it != dyn_table.end() &&
                                      static_cast<int>(it->second.hit_count) >= min_hits &&
                                      static_cast<double>(it->second.dyn_score) > gate;
              if (is_dynamic) {
                dynamic_points.push_back(p);
              } else {
                static_points.push_back(p);
              }
            }
            frame_static_publisher->publish(utils::eigen_to_point_cloud2(static_points, header));
            frame_dynamic_publisher->publish(utils::eigen_to_point_cloud2(dynamic_points, header));
          }
        }
        publish_odometry(lio->lidar_state, end_stamp);
        if (publish_lidar_acceleration) {
          publish_lidar_accel(lio->lidar_state.linear_acceleration, end_stamp);
        }
        if (publish_map_to_odom_tf && lio->config.camera_enabled) {
          publish_map_to_odom(end_stamp);
        }
      }
    } catch (const std::invalid_argument& ex) {
      RCLCPP_ERROR_STREAM(node->get_logger(), "Encountered error, dropping frame. Error: " << ex.what());
    }
  }
  atomic_node_running = false;
}

void Node::publish_odometry(const core::State& state, const core::Secondsd& stamp) const {
  const std::string_view from_frame = base_frame;
  const std::string_view to_frame = odom_frame;
  // tf message
  geometry_msgs::msg::TransformStamped transform_msg;
  transform_msg.header.stamp = rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(stamp).count());
  if (invert_odom_tf) {
    transform_msg.header.frame_id = from_frame;
    transform_msg.child_frame_id = to_frame;
    transform_msg.transform = utils::sophus_to_transform(state.pose.inverse());
  } else {
    transform_msg.header.frame_id = to_frame;
    transform_msg.child_frame_id = from_frame;
    transform_msg.transform = utils::sophus_to_transform(state.pose);
  }
  tf_broadcaster->sendTransform(transform_msg);

  // odometry msg
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(stamp).count());
  odom_msg.header.frame_id = to_frame;
  odom_msg.child_frame_id = from_frame;
  odom_msg.pose.pose = utils::sophus_to_pose(state.pose);
  utils::eigen_vector3d_to_ros_xyz(state.velocity, odom_msg.twist.twist.linear);
  utils::eigen_vector3d_to_ros_xyz(state.angular_velocity, odom_msg.twist.twist.angular);
  odom_publisher->publish(odom_msg);
}

void Node::publish_map_to_odom(const core::Secondsd& stamp) const {
  // REP-105: `map -> odom` carries the cumulative camera-keyframe correction
  // built up inside `LIO`. Defaults to identity until a keyframe lands, so
  // `map` is just an alias for `odom` for any downstream consumer until the
  // camera path produces a correction.
  const Sophus::SE3d& delta = lio->map_to_odom();
  geometry_msgs::msg::TransformStamped transform_msg;
  transform_msg.header.stamp = rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(stamp).count());
  transform_msg.header.frame_id = map_frame;
  transform_msg.child_frame_id = odom_frame;
  transform_msg.transform = utils::sophus_to_transform(delta);
  tf_broadcaster->sendTransform(transform_msg);
}

void Node::publish_lidar_accel(const Eigen::Vector3d& acceleration, const core::Secondsd& stamp) const {
  auto accel_msg = geometry_msgs::msg::AccelStamped();
  accel_msg.header.stamp = rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(stamp).count());
  accel_msg.header.frame_id = base_frame;
  utils::eigen_vector3d_to_ros_xyz(acceleration, accel_msg.accel.linear);
  lidar_accel_publisher->publish(accel_msg);
}

void Node::publish_map_loop() {
  while (atomic_node_running) {
    std::this_thread::sleep_for(publish_map_after);
    std::unique_lock lock(local_map_mutex);
    if (lio->map.Empty()) {
      RCLCPP_WARN_ONCE(node->get_logger(), "Local map publish thread: Local map is empty.");
      continue;
    }
    const core::Vector3dVector map_points = lio->map.Pointcloud();
    lock.unlock(); // we don't access the local map anymore
    std_msgs::msg::Header map_header;
    map_header.stamp = node->now();
    map_header.frame_id = odom_frame;
    map_publisher->publish(utils::eigen_to_point_cloud2(map_points, map_header));
  }
}

Node::~Node() {
  atomic_node_running = false;
  sync_condition_variable.notify_all();
}

void Node::dump_results_to_disk(const std::filesystem::path& results_dir, const std::string& run_name) const {
  try {
    std::filesystem::create_directories(results_dir); // no error if exists
    int index = 0;
    std::filesystem::path output_dir = results_dir / (run_name + "_" + std::to_string(index));
    while (std::filesystem::exists(output_dir)) {
      ++index;
      output_dir = results_dir / (run_name + "_" + std::to_string(index));
    }
    std::filesystem::create_directory(output_dir);
    const std::filesystem::path output_file = output_dir / (run_name + "_tum_" + std::to_string(index) + ".txt");
    // dump poses
    if (std::ofstream file(output_file); file.is_open()) {
      for (const auto& [timestamp, pose] : lio->poses_with_timestamps) {
        const Eigen::Vector3d& translation = pose.translation();
        const Eigen::Quaterniond& quaternion = pose.so3().unit_quaternion();
        file << std::fixed << std::setprecision(6) << timestamp.count() << " " << translation.x() << " "
             << translation.y() << " " << translation.z() << " " << quaternion.x() << " " << quaternion.y() << " "
             << quaternion.z() << " " << quaternion.w() << "\n";
      }
      std::cout << "Poses written to " << std::filesystem::absolute(output_file) << "\n";
    }
    // dump config
    const nlohmann::json json_config = {{"config", lio->config}};
    const std::filesystem::path config_file = output_dir / "config.json";
    if (std::ofstream file(config_file); file.is_open()) {
      file << json_config.dump(4);
      std::cout << "Configuration written to " << config_file << "\n";
    }
  } catch (const std::filesystem::filesystem_error& ex) {
    std::cerr << "[WARNING] Cannot write files to disk, encountered filesystem error: " << ex.what() << "\n";
  }
}

} // namespace rko_lio::ros
