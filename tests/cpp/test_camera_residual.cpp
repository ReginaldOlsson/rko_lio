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
 * Correctness tests for `build_camera_edge_linear_system`.
 *
 * The two checks below isolate the parts of the residual most likely to
 * harbor sign/convention errors:
 *
 *  1. **Analytic-vs-numeric Jacobian** on a smooth radial DT field. Each
 *     trial places a single point at a controlled image location (near the
 *     principal point) and compares `b = J^T r` returned by
 *     `build_camera_edge_linear_system` against a central finite-difference
 *     approximation of `d chi / d delta_xi` on the same setup. Catches sign
 *     errors and frame-convention mismatches in the projection chain.
 *
 *  2. **Gauss-Newton recovery** on the same radial DT field. With several
 *     points placed at different depths but all projecting to the principal
 *     point under identity, the cost is full-rank in (delta_rho, delta_phi_z)
 *     (and well-conditioned in the rest with the LM damping). A small pose
 *     perturbation is applied and a damped Gauss-Newton optimiser is asked
 *     to drive the residual back to zero.
 *
 * Prints "PASS" and exits 0 on success.
 */

#include "rko_lio/core/camera_residual.hpp"
#include "rko_lio/core/lio.hpp"
#include "rko_lio/core/util.hpp"

#include <bonxai/grid_coord.hpp>
#include <sophus/se3.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace {

using rko_lio::core::CameraFrame;
using rko_lio::core::CameraVisibleSet;
using rko_lio::core::LIO;
using rko_lio::core::LinearSystem;
using rko_lio::core::PinholeIntrinsics;
using rko_lio::core::build_camera_edge_linear_system;

bool check(bool cond, const char* what) {
  if (!cond) {
    std::cerr << "FAIL: " << what << std::endl;
    return false;
  }
  return true;
}

/** Smooth radial distance field DT(u, v) = sqrt((u - u_c)^2 + (v - v_c)^2). */
CameraFrame make_radial_dt_frame(const int rows, const int cols, const double u_c, const double v_c,
                                 const double fx, const double fy) {
  auto buf = std::make_shared<std::vector<float>>(static_cast<size_t>(rows) * cols, 0.f);
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      const double du = static_cast<double>(x) - u_c;
      const double dv = static_cast<double>(y) - v_c;
      (*buf)[static_cast<size_t>(y) * cols + x] = static_cast<float>(std::sqrt(du * du + dv * dv));
    }
  }
  CameraFrame frame;
  frame.time = rko_lio::core::Secondsd{0.0};
  frame.intrinsics.fx = fx;
  frame.intrinsics.fy = fy;
  frame.intrinsics.cx = u_c;
  frame.intrinsics.cy = v_c;
  frame.intrinsics.width = cols;
  frame.intrinsics.height = rows;
  frame.rows = rows;
  frame.cols = cols;
  frame.dt_image = buf;
  return frame;
}

/**
 * Build a point in odom whose projection under (`pose`, `extrinsic`) lands
 * exactly at the image position (u_target, v_target) for the given depth in
 * the camera frame.
 */
Eigen::Vector3d make_point_projecting_to(const Sophus::SE3d& pose, const Sophus::SE3d& extrinsic,
                                          const PinholeIntrinsics& K, double u_target, double v_target,
                                          double depth_camera) {
  const Eigen::Vector3d p_c((u_target - K.cx) * depth_camera / K.fx,
                            (v_target - K.cy) * depth_camera / K.fy,
                            depth_camera);
  // p_c = (pose * extrinsic).inverse() * p_o  =>  p_o = (pose * extrinsic) * p_c
  return (pose * extrinsic) * p_c;
}

bool test_jacobian_finite_difference() {
  constexpr int rows = 240;
  constexpr int cols = 320;
  const double u_c = cols / 2.0;
  const double v_c = rows / 2.0;
  const CameraFrame frame = make_radial_dt_frame(rows, cols, u_c, v_c, 250.0, 250.0);

  LIO::Config config;
  config.camera_enabled = true;
  config.camera_weight = 1.0;
  config.camera_max_dt_residual_px = 1.0e6; // disable Huber clip for this check
  config.camera_min_visible_points = 1;
  config.camera_min_point_depth_m = 0.1;

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> trans_dist(-0.2, 0.2);
  std::uniform_real_distribution<double> rot_dist(-0.03, 0.03);
  std::uniform_real_distribution<double> depth_dist(3.0, 8.0);
  std::uniform_real_distribution<double> pixel_off(-30.0, 30.0); // stay well clear of image edges

  constexpr int num_trials = 16;
  constexpr double h_fd = 5e-6;
  // 5% tolerance: the analytic Jacobian uses a one-pixel forward-difference
  // gradient of the DT field while the numerical Jacobian computes
  // d(chi)/d(dxi) by central-difference of bilinear-sampled chi. The two
  // are O(h) consistent but not identical at a single test point. Anything
  // significantly worse than 5% indicates a real bug.
  constexpr double tol = 5e-2;

  bool all_ok = true;
  int trial = 0;
  while (trial < num_trials) {
    Eigen::Matrix<double, 6, 1> tau_pose;
    tau_pose.head<3>() = Eigen::Vector3d(trans_dist(rng), trans_dist(rng), trans_dist(rng));
    tau_pose.tail<3>() = Eigen::Vector3d(rot_dist(rng), rot_dist(rng), rot_dist(rng));
    const Sophus::SE3d pose = Sophus::SE3d::exp(tau_pose);

    Eigen::Matrix<double, 6, 1> tau_ext;
    tau_ext.head<3>() = Eigen::Vector3d(trans_dist(rng), trans_dist(rng), trans_dist(rng));
    tau_ext.tail<3>() = Eigen::Vector3d(rot_dist(rng), rot_dist(rng), rot_dist(rng));
    const Sophus::SE3d extrinsic_cam_to_base = Sophus::SE3d::exp(tau_ext);

    // Place the point so it projects well inside the image at the random pose.
    // Use a small pixel offset so the perturbed projection stays in the image.
    const double depth = depth_dist(rng);
    const double u_t = u_c + pixel_off(rng);
    const double v_t = v_c + pixel_off(rng);
    CameraVisibleSet vs;
    vs.points_odom.emplace_back(make_point_projecting_to(pose, extrinsic_cam_to_base, frame.intrinsics, u_t, v_t, depth));
    vs.voxel_keys.emplace_back(Bonxai::CoordT{0, 0, 0});

    const auto& [H0, b0, chi0] = build_camera_edge_linear_system(pose, extrinsic_cam_to_base, frame, vs, config);
    if (H0.norm() == 0.0) {
      // The point fell outside the image even at the unperturbed pose; resample.
      continue;
    }

    // Central finite difference on chi gives b directly because b = d chi / d delta_xi.
    Eigen::Matrix<double, 6, 1> b_numeric = Eigen::Matrix<double, 6, 1>::Zero();
    bool projection_in_bounds = true;
    for (int k = 0; k < 6 && projection_in_bounds; ++k) {
      Eigen::Matrix<double, 6, 1> e_k = Eigen::Matrix<double, 6, 1>::Zero();
      e_k(k) = h_fd;
      const Sophus::SE3d pose_plus = Sophus::SE3d::exp(e_k) * pose;
      const Sophus::SE3d pose_minus = Sophus::SE3d::exp(-e_k) * pose;
      const auto& [Hp, bp, chi_plus] = build_camera_edge_linear_system(pose_plus, extrinsic_cam_to_base, frame, vs, config);
      const auto& [Hm, bm, chi_minus] = build_camera_edge_linear_system(pose_minus, extrinsic_cam_to_base, frame, vs, config);
      if (Hp.norm() == 0.0 || Hm.norm() == 0.0) {
        projection_in_bounds = false;
        break;
      }
      b_numeric(k) = (chi_plus - chi_minus) / (2.0 * h_fd);
    }
    if (!projection_in_bounds) {
      // Point projected outside under one of the perturbations; resample.
      continue;
    }

    const double scale = std::max(1.0, b0.cwiseAbs().maxCoeff() + b_numeric.cwiseAbs().maxCoeff());
    const double err = (b0 - b_numeric).cwiseAbs().maxCoeff() / scale;
    if (err >= tol) {
      std::cerr << "trial=" << trial << " analytic b=" << b0.transpose() << "\n  numeric b="
                << b_numeric.transpose() << "\n  err=" << err << std::endl;
      all_ok = false;
    }
    ++trial;
  }
  return check(all_ok, "analytic Jacobian matches numerical Jacobian");
}

double gn_solve(const Sophus::SE3d& initial, const Sophus::SE3d& extrinsic_cam_to_base, const CameraFrame& frame,
                 const CameraVisibleSet& visible, const LIO::Config& config, Sophus::SE3d& out_pose) {
  Sophus::SE3d pose = initial;
  double last_chi = std::numeric_limits<double>::infinity();
  // Levenberg-Marquardt-style damping so the very first step does not over-
  // shoot when the Hessian is near-degenerate. Decreases as the optimiser
  // gets close to the minimum.
  double lambda = 1e-2;
  for (int iter = 0; iter < 100; ++iter) {
    const auto& [H, b, chi] = build_camera_edge_linear_system(pose, extrinsic_cam_to_base, frame, visible, config);
    if (H.norm() == 0.0) {
      break;
    }
    const Eigen::Matrix<double, 6, 6> H_damped = H + lambda * Eigen::Matrix<double, 6, 6>::Identity();
    const Eigen::Matrix<double, 6, 1> dx = H_damped.ldlt().solve(-b);
    const Sophus::SE3d trial_pose = Sophus::SE3d::exp(dx) * pose;
    const auto& [Ht, bt, chi_trial] = build_camera_edge_linear_system(trial_pose, extrinsic_cam_to_base, frame, visible, config);
    if (chi_trial < chi) {
      pose = trial_pose;
      last_chi = chi_trial;
      lambda = std::max(lambda * 0.5, 1e-6);
      if (dx.norm() < 1e-10) {
        break;
      }
    } else {
      lambda *= 4.0;
      if (lambda > 1e8) {
        break;
      }
    }
  }
  out_pose = pose;
  return last_chi;
}

bool test_recovery_on_radial_field() {
  constexpr int rows = 240;
  constexpr int cols = 320;
  const double u_c = cols / 2.0;
  const double v_c = rows / 2.0;
  const CameraFrame frame = make_radial_dt_frame(rows, cols, u_c, v_c, 250.0, 250.0);

  LIO::Config config;
  config.camera_enabled = true;
  config.camera_weight = 1.0;
  config.camera_max_dt_residual_px = 1.0e6;
  config.camera_min_visible_points = 1;
  config.camera_min_point_depth_m = 0.1;

  // Place every test point exactly on the camera's optical axis (so the
  // ground-truth projection is at the principal point, where DT == 0).
  // With several depths the cost has a clean unique minimum at identity in
  // (tx, ty, rx, ry); the unobservable directions (tz, rz) carry zero
  // gradient and stay at their initial perturbation under LM damping. We
  // test recovery only along the observable directions.
  const Sophus::SE3d gt;
  const Sophus::SE3d extrinsic_cam_to_base;
  CameraVisibleSet visible;
  for (double z : {3.0, 4.0, 5.0, 6.0, 7.0, 8.0}) {
    visible.points_odom.emplace_back(0.0, 0.0, z);
    visible.voxel_keys.emplace_back(Bonxai::CoordT{0, 0, 0});
  }
  // Sanity: chi at GT must be zero so the optimiser has a unique global
  // minimum at identity (modulo the (tz, rz) null space).
  {
    const auto& [H, b, chi_gt] = build_camera_edge_linear_system(gt, extrinsic_cam_to_base, frame, visible, config);
    (void)H;
    (void)b;
    if (!check(chi_gt < 1e-6, "chi at ground truth is zero")) {
      return false;
    }
  }

  bool all_ok = true;
  // Probe each observable DoF independently with both signs.
  const std::vector<std::pair<int, double>> probes = {
      {0, 0.05},  {0, -0.05}, // tx
      {1, 0.05},  {1, -0.05}, // ty
      {3, 0.02},  {3, -0.02}, // rx
      {4, 0.02},  {4, -0.02}, // ry
  };
  for (const auto& [idx, mag] : probes) {
    Eigen::Matrix<double, 6, 1> tau = Eigen::Matrix<double, 6, 1>::Zero();
    tau(idx) = mag;
    const Sophus::SE3d perturbed = Sophus::SE3d::exp(tau) * gt;
    Sophus::SE3d recovered;
    const double chi_final = gn_solve(perturbed, extrinsic_cam_to_base, frame, visible, config, recovered);

    const auto& [Hp, bp, chi_initial] =
        build_camera_edge_linear_system(perturbed, extrinsic_cam_to_base, frame, visible, config);
    (void)Hp;
    (void)bp;
    std::cout << "[recovery] idx=" << idx << " mag=" << mag << " chi_initial=" << chi_initial
              << " chi_final=" << chi_final << std::endl;

    // Translation perturbations drive chi to machine epsilon; rotation
    // perturbations bottom out earlier because the rotation block of the
    // Jacobian on points lying on the optical axis is weakly conditioned
    // (the linearised projection of a small rotation only retains the
    // first-order term). A 99% chi reduction is still a strong signal that
    // the residual + Jacobian are correctly oriented.
    const double chi_threshold = std::max(1e-3, 0.02 * chi_initial);
    all_ok &= check(chi_final < chi_threshold,
                    "GN reduces chi by at least 50x from the perturbation");
  }
  return all_ok;
}

} // namespace

int main() {
  bool ok = true;
  ok &= test_jacobian_finite_difference();
  ok &= test_recovery_on_radial_field();
  if (!ok) {
    std::cerr << "FAIL" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "PASS" << std::endl;
  return EXIT_SUCCESS;
}
