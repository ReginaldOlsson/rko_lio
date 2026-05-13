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
 * @file camera_residual.hpp
 * Camera edge-alignment residual block for tight coupling of a monocular
 * camera into the per-scan Gauss-Newton solver in lio.cpp.
 *
 * The math is documented in docs/pages/camera_coupling.rst.
 *
 * This translation unit deliberately does not depend on OpenCV: the
 * distance-transform image is consumed via a raw float buffer owned by
 * the CameraFrame, and Canny + cv::distanceTransform are computed in the
 * ROS (or Python) layer instead.
 */

#pragma once
#include "lio.hpp"
#include "sparse_voxel_grid.hpp"
#include "util.hpp"
#include <bonxai/grid_coord.hpp>
#include <sophus/se3.hpp>
#include <unordered_map>
#include <vector>

namespace rko_lio::core {

/**
 * Pre-projected list of representative map points to evaluate the camera
 * residual at during ICP. Computed once per scan from the predicted base
 * pose, then reused across all Gauss-Newton iterations to keep the inner
 * loop cheap.
 *
 * `points_odom[i]` is the representative 3D position of voxel
 * `voxel_keys[i]` in the odom frame. Dynamic-flagged voxels are excluded
 * from this set when the dynamic-segmentation feature is enabled.
 */
struct CameraVisibleSet {
  std::vector<Eigen::Vector3d> points_odom;
  std::vector<Bonxai::CoordT> voxel_keys;
};

/**
 * Build a `CameraVisibleSet` by iterating the voxel grid, projecting each
 * voxel's representative point with the predicted pose, and culling points
 * that are behind the camera, too close to the camera, outside the image
 * rectangle, or whose voxel has been flagged dynamic. An optional low-res
 * z-buffer occlusion check keeps only the foreground sample per pixel.
 *
 * @param voxel_map                  The current local map.
 * @param predicted_base_to_odom     Predicted pose at the image time.
 * @param extrinsic_cam_to_base      Fixed camera->base extrinsic.
 * @param frame                      Camera frame (provides intrinsics + rect).
 * @param config                     LIO config (provides camera_* gates).
 * @param voxel_dyn                  Optional side-table for dynamic masking;
 *                                   pass `nullptr` to disable masking.
 */
CameraVisibleSet
compute_camera_visible_set(const SparseVoxelGrid& voxel_map,
                           const Sophus::SE3d& predicted_base_to_odom,
                           const Sophus::SE3d& extrinsic_cam_to_base,
                           const CameraFrame& frame,
                           const LIO::Config& config,
                           const std::unordered_map<Bonxai::CoordT, VoxelDynStats>* voxel_dyn);

/**
 * Build the camera edge-alignment linear system at `current_pose` over the
 * pre-projected visible set. Returns (H, b, chi) summed into the same form
 * used by build_icp_linear_system and build_orientation_linear_system, with
 * Huber weighting and `1 / N_vis` normalization applied internally. The
 * result is additionally scaled by `config.camera_weight`.
 *
 * If `visible.points_odom.size() < config.camera_min_visible_points`, the
 * returned LinearSystem is zero (camera contributes nothing for this scan).
 */
LinearSystem build_camera_edge_linear_system(const Sophus::SE3d& current_pose,
                                              const Sophus::SE3d& extrinsic_cam_to_base,
                                              const CameraFrame& frame,
                                              const CameraVisibleSet& visible,
                                              const LIO::Config& config);

/**
 * Bundle of inputs that `icp(...)` needs to evaluate the camera edge
 * residual at each Gauss-Newton iteration. The visible set is computed once
 * per scan from the predicted pose; the rest are constant per scan.
 *
 * Passing a `nullptr` for this context (or leaving any sub-pointer null)
 * disables the camera block. Mirrors the pattern of `DynWeightingContext`.
 */
struct CameraResidualContext {
  const CameraFrame* frame = nullptr;
  const Sophus::SE3d* extrinsic_cam_to_base = nullptr;
  const CameraVisibleSet* visible = nullptr;
};

} // namespace rko_lio::core
