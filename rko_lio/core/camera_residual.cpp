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

#include "camera_residual.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rko_lio::core {

namespace {

/**
 * Sample a row-major float field bilinearly at fractional pixel (u, v) and
 * simultaneously return the exact analytic gradient of the bilinear
 * interpolant in the enclosing unit cell.
 *
 * The bilinear interpolant in cell [u0, u0+1] x [v0, v0+1] is
 *     f_bi(u, v) = (1-a)(1-b) f00 + a(1-b) f10 + (1-a)b f01 + a*b f11
 * with a = u - u0, b = v - v0. Its partials are piecewise constant per cell,
 * not biased by which side of a DT kink the query lies on, which matters for
 * Gauss-Newton stability near the field's minima where forward differences
 * collapse to ~0 and produce wildly underestimated step sizes.
 */
struct BilinearValueGrad {
  double value;
  Eigen::RowVector2d grad; // d/du, d/dv
};

inline BilinearValueGrad bilinear_sample_and_grad(const float* data, int cols, int rows, double u, double v) {
  const double uc = std::clamp(u, 0.0, static_cast<double>(cols - 1));
  const double vc = std::clamp(v, 0.0, static_cast<double>(rows - 1));
  const int u0 = static_cast<int>(std::floor(uc));
  const int v0 = static_cast<int>(std::floor(vc));
  const int u1 = std::min(u0 + 1, cols - 1);
  const int v1 = std::min(v0 + 1, rows - 1);
  const double a = uc - static_cast<double>(u0);
  const double b = vc - static_cast<double>(v0);
  const double f00 = data[v0 * cols + u0];
  const double f10 = data[v0 * cols + u1];
  const double f01 = data[v1 * cols + u0];
  const double f11 = data[v1 * cols + u1];
  const double value = (1.0 - a) * (1.0 - b) * f00 + a * (1.0 - b) * f10 + (1.0 - a) * b * f01 + a * b * f11;
  const double df_du = (1.0 - b) * (f10 - f00) + b * (f11 - f01);
  const double df_dv = (1.0 - a) * (f01 - f00) + a * (f11 - f10);
  return {value, Eigen::RowVector2d(df_du, df_dv)};
}

inline float bilinear_sample(const float* data, int cols, int rows, double u, double v) {
  return static_cast<float>(bilinear_sample_and_grad(data, cols, rows, u, v).value);
}

/**
 * Huber kernel: returns the per-residual weight `rho'(r) / r` style scalar so
 * that the linear system `H = sum w_i^2 J_i^T J_i` and `b = sum w_i^2 J_i^T r_i`
 * is equivalent to a robust-cost Gauss-Newton step.
 */
inline double huber_weight(double r, double k) {
  const double abs_r = std::abs(r);
  if (abs_r <= k) {
    return 1.0;
  }
  return k / abs_r;
}

/** Centroid of a voxel block; the representative point used in projection. */
inline Eigen::Vector3d voxel_centroid(const VoxelBlock& block) {
  Eigen::Vector3d c = Eigen::Vector3d::Zero();
  for (const auto& p : block) {
    c += p;
  }
  c /= static_cast<double>(block.size());
  return c;
}

} // namespace

CameraVisibleSet
compute_camera_visible_set(const SparseVoxelGrid& voxel_map,
                           const Sophus::SE3d& predicted_base_to_odom,
                           const Sophus::SE3d& extrinsic_cam_to_base,
                           const CameraFrame& frame,
                           const LIO::Config& config,
                           const std::unordered_map<Bonxai::CoordT, VoxelDynStats>* voxel_dyn) {
  CameraVisibleSet result;
  if (frame.intrinsics.width <= 0 || frame.intrinsics.height <= 0) {
    return result;
  }

  const Sophus::SE3d odom_to_cam = (predicted_base_to_odom * extrinsic_cam_to_base).inverse();
  const double fx = frame.intrinsics.fx;
  const double fy = frame.intrinsics.fy;
  const double cx = frame.intrinsics.cx;
  const double cy = frame.intrinsics.cy;
  const int W = frame.intrinsics.width;
  const int H = frame.intrinsics.height;
  const double z_min = config.camera_min_point_depth_m;

  // Reserve roughly based on number of active voxels; bonxai gives us a count.
  result.points_odom.reserve(voxel_map.map_.activeCellsCount());
  result.voxel_keys.reserve(voxel_map.map_.activeCellsCount());

  // Optional dynamic mask gate parameters.
  const double dyn_gate = config.dyn_skip_map_score;
  const int dyn_min_hits = config.dyn_min_hits_to_trust;
  const bool apply_dyn_mask = voxel_dyn != nullptr;

  // Optional z-buffer for soft occlusion. Stored in low-res grid.
  const int zb_down = std::max(1, config.camera_zbuf_downsample);
  const int zb_W = std::max(1, W / zb_down);
  const int zb_H = std::max(1, H / zb_down);
  std::vector<float> zbuf;
  if (config.camera_use_occlusion_zbuf) {
    zbuf.assign(static_cast<size_t>(zb_W) * static_cast<size_t>(zb_H), std::numeric_limits<float>::infinity());
  }

  // Cache per-candidate metadata so we can do a second pass for occlusion
  // without re-projecting every voxel.
  struct Candidate {
    Eigen::Vector3d p_o;
    Bonxai::CoordT key;
    double u;
    double v;
    double depth;
    int zb_idx;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(voxel_map.map_.activeCellsCount());

  voxel_map.map_.forEachCell([&](const VoxelBlock& block, const Bonxai::CoordT& key) {
    if (block.empty()) {
      return;
    }
    if (apply_dyn_mask) {
      const auto it = voxel_dyn->find(key);
      if (it != voxel_dyn->end() && static_cast<int>(it->second.hit_count) >= dyn_min_hits &&
          static_cast<double>(it->second.dyn_score) > dyn_gate) {
        return; // skip dynamic-flagged voxel
      }
    }
    const Eigen::Vector3d p_o = voxel_centroid(block);
    const Eigen::Vector3d p_c = odom_to_cam * p_o;
    if (p_c.z() < z_min) {
      return;
    }
    const double u = fx * p_c.x() / p_c.z() + cx;
    const double v = fy * p_c.y() / p_c.z() + cy;
    if (u < 0.0 || u >= static_cast<double>(W) || v < 0.0 || v >= static_cast<double>(H)) {
      return;
    }
    int zb_idx = -1;
    if (config.camera_use_occlusion_zbuf) {
      const int zu = std::min(zb_W - 1, static_cast<int>(u) / zb_down);
      const int zv = std::min(zb_H - 1, static_cast<int>(v) / zb_down);
      zb_idx = zv * zb_W + zu;
      const float z = static_cast<float>(p_c.z());
      if (z < zbuf[zb_idx]) {
        zbuf[zb_idx] = z;
      }
    }
    candidates.push_back({p_o, key, u, v, p_c.z(), zb_idx});
  });

  if (!config.camera_use_occlusion_zbuf) {
    result.points_odom.reserve(candidates.size());
    result.voxel_keys.reserve(candidates.size());
    for (const auto& c : candidates) {
      result.points_odom.push_back(c.p_o);
      result.voxel_keys.push_back(c.key);
    }
    return result;
  }

  // Second pass: keep only candidates within a tolerance of the foreground.
  // Tolerance scales with depth so distant points get more slack.
  for (const auto& c : candidates) {
    const float front = zbuf[c.zb_idx];
    const double tol = std::max(0.5, 0.05 * c.depth); // 5% of depth, at least 50 cm
    if (c.depth - static_cast<double>(front) <= tol) {
      result.points_odom.push_back(c.p_o);
      result.voxel_keys.push_back(c.key);
    }
  }
  return result;
}

LinearSystem build_camera_edge_linear_system(const Sophus::SE3d& current_pose,
                                              const Sophus::SE3d& extrinsic_cam_to_base,
                                              const CameraFrame& frame,
                                              const CameraVisibleSet& visible,
                                              const LIO::Config& config) {
  Eigen::Matrix6d H = Eigen::Matrix6d::Zero();
  Eigen::Vector6d b = Eigen::Vector6d::Zero();
  double chi = 0.0;

  if (!frame.dt_image || visible.points_odom.empty()) {
    return {H, b, chi};
  }
  if (frame.rows <= 0 || frame.cols <= 0 ||
      static_cast<int>(frame.dt_image->size()) != frame.rows * frame.cols) {
    return {H, b, chi};
  }
  if (static_cast<int>(visible.points_odom.size()) < config.camera_min_visible_points) {
    return {H, b, chi};
  }

  // T_odom2cam = (current_pose * extrinsic_cam_to_base).inverse(); only the
  // rotation enters the per-point Jacobian (the constant translation cancels
  // because the perturbation is applied to T, not to p_o).
  const Sophus::SE3d odom_to_cam = (current_pose * extrinsic_cam_to_base).inverse();
  const Eigen::Matrix3d R_odom_to_cam = odom_to_cam.rotationMatrix();
  const double fx = frame.intrinsics.fx;
  const double fy = frame.intrinsics.fy;
  const double cx = frame.intrinsics.cx;
  const double cy = frame.intrinsics.cy;
  const int cols = frame.cols;
  const int rows = frame.rows;
  const float* data = frame.dt_image->data();
  const double huber_k = std::max(1.0, 0.5 * config.camera_max_dt_residual_px);
  const double r_clip = config.camera_max_dt_residual_px;

  int n_used = 0;
  for (const auto& p_o : visible.points_odom) {
    const Eigen::Vector3d p_c = odom_to_cam * p_o;
    if (p_c.z() < config.camera_min_point_depth_m) {
      continue;
    }
    const double inv_z = 1.0 / p_c.z();
    const double u = fx * p_c.x() * inv_z + cx;
    const double v = fy * p_c.y() * inv_z + cy;
    if (u < 0.0 || u >= static_cast<double>(cols) || v < 0.0 || v >= static_cast<double>(rows)) {
      continue;
    }

    const BilinearValueGrad vg = bilinear_sample_and_grad(data, cols, rows, u, v);
    const double r = std::clamp(vg.value, -r_clip, r_clip);
    const Eigen::RowVector2d grad_dt = vg.grad;

    // du/dp_c (2x3) for the pinhole model.
    Eigen::Matrix<double, 2, 3> du_dpc;
    du_dpc(0, 0) = fx * inv_z;
    du_dpc(0, 1) = 0.0;
    du_dpc(0, 2) = -fx * p_c.x() * inv_z * inv_z;
    du_dpc(1, 0) = 0.0;
    du_dpc(1, 1) = fy * inv_z;
    du_dpc(1, 2) = -fy * p_c.y() * inv_z * inv_z;

    // Left-perturbation Jacobian in odom: T <- exp(delta_xi) T, so
    //   p_c = R_odom_to_cam * (exp(-delta_xi) * p_o) + const,
    // giving  dp_c/d_delta_xi = R_odom_to_cam * [-I, [p_o]_x].
    // This matches the convention used by build_icp_linear_system (where the
    // rotation block uses -[source]_x with source = T*p_scan in odom).
    Eigen::Matrix<double, 3, 6> dpc_ddxi;
    dpc_ddxi.block<3, 3>(0, 0) = -R_odom_to_cam;
    dpc_ddxi.block<3, 3>(0, 3) = R_odom_to_cam * Sophus::SO3d::hat(p_o);

    const Eigen::Matrix<double, 1, 6> J = grad_dt * du_dpc * dpc_ddxi;
    const double w = huber_weight(r, huber_k);
    const double w2 = w * w;
    H.noalias() += w2 * J.transpose() * J;
    b.noalias() += w2 * J.transpose() * r;
    chi += 0.5 * w2 * r * r;
    ++n_used;
  }

  if (n_used < config.camera_min_visible_points) {
    return {Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero(), 0.0};
  }
  const double alpha = config.camera_weight / static_cast<double>(n_used);
  return {alpha * H, alpha * b, chi};
}

} // namespace rko_lio::core
