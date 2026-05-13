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
#include "rko_lio/core/lio.hpp"
#include "rko_lio/core/process_timestamps.hpp"
// stl
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
// ros
#include <geometry_msgs/msg/accel_stamped.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

namespace rko_lio::ros {
class Node {
public:
  rclcpp::Node::SharedPtr node;
  std::unique_ptr<core::LIO> lio;
  core::TimestampProcessingConfig timestamp_proc_config;

  std::string imu_topic;
  std::string imu_frame = ""; // default: get from the first imu message
  std::string lidar_topic;
  std::string lidar_frame = ""; // default: get from the first lidar message
  std::string base_frame;
  std::string odom_frame = "odom";
  std::string map_frame = "map";
  std::string odom_topic = "rko_lio/odometry";
  std::string map_topic = "rko_lio/local_map";
  std::string deskewed_scan_topic = "rko_lio/frame";

  // ---- camera tight-coupling (opt-in via lio config.camera_enabled) ----
  std::string image_topic = "image";
  std::string camera_info_topic = "camera_info";
  std::string camera_frame = "";          // default: get from camera_info / image header
  double canny_low_threshold = 50.0;
  double canny_high_threshold = 150.0;
  int canny_aperture_size = 3;
  double image_max_rate_hz = 30.0;
  bool publish_map_to_odom_tf = true;

  bool dump_results = false;
  std::string results_dir = "results";
  std::string run_name = "rko_lio_run";

  bool invert_odom_tf = false;
  bool publish_lidar_acceleration = false;
  bool publish_deskewed_scan = false;
  bool publish_local_map = false;
  /** When `publish_deskewed_scan` is on, also split the cloud into static/dynamic. */
  bool publish_dynamic_split = false;
  std::string deskewed_scan_static_topic = "rko_lio/frame_static";
  std::string deskewed_scan_dynamic_topic = "rko_lio/frame_dynamic";

  Sophus::SE3d extrinsic_imu2base;
  Sophus::SE3d extrinsic_lidar2base;
  bool extrinsics_set = false;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frame_publisher;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frame_static_publisher;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frame_dynamic_publisher;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher;
  rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr lidar_accel_publisher;

  // Camera pipeline. Latency is dominated by Canny + cv::distanceTransform on
  // the image-callback thread; the registration loop only drains the queue.
  image_transport::Subscriber image_subscriber;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub;
  std::mutex camera_intrinsics_mutex;
  bool camera_intrinsics_ready = false;
  core::PinholeIntrinsics camera_intrinsics;
  cv::Mat rectify_map_x;
  cv::Mat rectify_map_y;
  bool needs_rectification = false;
  core::Secondsd last_image_processed_time{0.0};

  // multithreading
  std::jthread map_publish_thead;
  core::Secondsd publish_map_after = std::chrono::seconds(1);
  std::mutex local_map_mutex;

  std::jthread registration_thread;
  std::mutex buffer_mutex;
  std::condition_variable sync_condition_variable;
  std::atomic<bool> atomic_node_running = true;
  std::atomic<bool> atomic_can_process = false;
  std::queue<core::ImuControl> imu_buffer;
  std::queue<core::LidarFrame> lidar_buffer;
  std::queue<core::CameraFrame> camera_buffer;
  size_t max_lidar_buffer_size = 50;
  size_t max_camera_buffer_size = 30;

  Node() = delete;
  Node(const std::string& node_name, const rclcpp::NodeOptions& options);

  void parse_cli_extrinsics();
  bool check_and_set_extrinsics();
  /** Try to resolve the camera->base extrinsic; safe to call repeatedly. */
  void try_set_camera_extrinsic();
  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg);
  void lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg);
  void camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg);
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg);
  void drain_camera_buffer_for_scan(const core::Secondsd& scan_end_time);
  void registration_loop();
  void publish_odometry(const core::State& state, const core::Secondsd& stamp) const;
  void publish_lidar_accel(const Eigen::Vector3d& acceleration, const core::Secondsd& stamp) const;
  void publish_map_loop();
  void publish_map_to_odom(const core::Secondsd& stamp) const;
  void dump_results_to_disk(const std::filesystem::path& results_dir, const std::string& run_name) const;

  ~Node();
  Node(const Node&) = delete;
  Node(Node&&) = delete;
  Node& operator=(const Node&) = delete;
  Node& operator=(Node&&) = delete;
};
} // namespace rko_lio::ros
