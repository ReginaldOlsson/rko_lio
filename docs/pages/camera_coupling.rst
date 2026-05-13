Camera tight coupling and dynamic-point segmentation
=====================================================

This page derives the two opt-in additions to the core LIO pipeline that share
plumbing in :doc:`./core` ``icp(...)``:

1. A camera **edge-alignment residual** that is summed into the per-scan
   Gauss-Newton system alongside the existing ICP residual.
2. A per-voxel **dynamic-point statistic** that is updated from the same ICP
   residuals and used to clean the map, downweight bad correspondences, and
   mask out dynamic regions from the camera projection.

Both features default to **off** so the bit-for-bit behavior of the published
RKO-LIO odometry is preserved unless they are explicitly enabled.

Conventions
-----------

We follow the same notation as the rest of the codebase: ``transform_from2to``
means a transformation that maps a vector expressed in the ``<from>`` frame to
the ``<to>`` frame, i.e. ``v_to = T_from2to * v_from``.

For the camera derivation:

- :math:`T = {}^{\mathrm{odom}}T_{\mathrm{base}}`: pose of the base frame in
  odom, the pose being optimized by ``icp``. ``current_pose`` in code.
- :math:`E = {}^{\mathrm{base}}T_{\mathrm{cam}}`: fixed extrinsic from camera
  to base, configured once from tf2 or a YAML.
- :math:`K`: pinhole intrinsic matrix
  :math:`\mathrm{diag}(f_x, f_y, 1)` with principal point :math:`(c_x, c_y)`.
  We use rectified images and rectified ``CameraInfo``; the lens distortion
  is handled in the ROS layer.
- :math:`\pi(p) = (p_x / p_z, p_y / p_z)` is the perspective projection.

ICP perturbs ``current_pose`` via a left-multiplied increment in ``odom``:
:math:`T \leftarrow \exp(\delta\xi) T`. This convention is fixed by
``build_icp_linear_system`` in ``rko_lio/core/lio.cpp`` and is reused for the
camera residual so the three Hessian blocks (ICP, orientation regularization,
camera) can be summed without sign reconciliation.

Camera edge-alignment residual
------------------------------

Image preprocessing (done **once per image** in the ROS layer, not per ICP
iteration):

1. Rectify with ``CameraInfo``.
2. Run Canny edge detection.
3. Compute the unsigned :math:`L_2` distance transform :math:`DT(u, v)`. For
   any pixel :math:`(u, v)`, :math:`DT(u, v)` is the distance to the nearest
   Canny edge, in pixels. It is smooth almost everywhere and admits a
   bilinear gradient :math:`\nabla DT(u, v)` away from the medial axis.

Per visible map point :math:`p_o \in \mathrm{odom}`:

.. math::

   p_c &= (E^{-1} T^{-1}) \, p_o
   = {}^{\mathrm{cam}}T_{\mathrm{odom}} \, p_o \\
   u   &= \pi(K p_c)
   = \left( f_x \frac{p_{c,x}}{p_{c,z}} + c_x,\;
            f_y \frac{p_{c,y}}{p_{c,z}} + c_y \right) \\
   r_i &= DT(u_i)

The residual is scalar, so each map point contributes a :math:`1 \times 6`
Jacobian :math:`J_i`.

Jacobian
^^^^^^^^

With left perturbation :math:`T \leftarrow \exp(\delta\xi)\,T` in the odom
frame (the same convention used by ``build_icp_linear_system`` in
``rko_lio/core/lio.cpp``), :math:`T^{-1} \leftarrow T^{-1}\,\exp(-\delta\xi)`
so the map point is left-multiplied by :math:`\exp(-\delta\xi)` before being
transformed into the camera. Define
:math:`R_{\mathrm{odom}\to\mathrm{cam}}
= R(E^{-1})\,R(T^{-1})
= R\!\left((T E)^{-1}\right)`.

Then:

.. math::

   \frac{\partial p_c}{\partial \delta\xi}
   = R_{\mathrm{odom}\to\mathrm{cam}}
   \begin{bmatrix} -I_{3 \times 3} & [p_o]_\times \end{bmatrix}

where :math:`[\cdot]_\times` is the skew-symmetric matrix and the layout is
:math:`\delta\xi = (\delta\rho, \delta\phi)` (translation first, rotation
second), matching the Sophus convention used in ``rko_lio/core/lio.cpp``.
The constant translation parts of :math:`E^{-1}` and :math:`T^{-1}` drop
out because the perturbation enters multiplicatively only on the rotation
of :math:`T^{-1}`.

The projection Jacobian for a pinhole camera is:

.. math::

   \frac{\partial u}{\partial p_c}
   = \frac{1}{p_{c,z}}
   \begin{bmatrix}
   f_x & 0   & -f_x \, p_{c,x} / p_{c,z} \\
   0   & f_y & -f_y \, p_{c,y} / p_{c,z}
   \end{bmatrix}

Combining:

.. math::

   J_i = \nabla DT(u_i) \cdot
         \frac{\partial u}{\partial p_c} \cdot
         \frac{\partial p_c}{\partial \delta\xi}
   \in \mathbb{R}^{1 \times 6}

with :math:`\nabla DT(u_i)` bilinearly sampled from the precomputed DT image.

The linear system contribution per point is

.. math::

   H_{\mathrm{cam},i} = J_i^\top J_i, \qquad
   b_{\mathrm{cam},i} = J_i^\top r_i.

Robust weighting
^^^^^^^^^^^^^^^^

Each point receives a Huber-style weight :math:`\rho(r_i)`. Defaults:

- Clip :math:`|r_i|` to ``camera_max_dt_residual_px`` (default 20 pixels) so
  catastrophic associations at cold start cannot dominate.
- Apply a Huber kernel with threshold equal to half the clip.

The aggregated system is normalized by the number of visible points and
multiplied by ``camera_weight`` :math:`\alpha`:

.. math::

   H_{\mathrm{cam}} = \frac{\alpha}{N_{\mathrm{vis}}}
   \sum_i \rho(r_i)^2 \, H_{\mathrm{cam},i},
   \qquad
   b_{\mathrm{cam}} = \frac{\alpha}{N_{\mathrm{vis}}}
   \sum_i \rho(r_i)^2 \, b_{\mathrm{cam},i}.

If :math:`N_{\mathrm{vis}} <` ``camera_min_visible_points`` the camera block
is disabled for this scan (yields ``H = 0, b = 0``).

Integration with ICP
^^^^^^^^^^^^^^^^^^^^

Inside ``icp(...)`` the per-iteration update becomes::

   auto [H_icp, b_icp, chi_icp] = build_icp_linear_system(...);
   auto [H_cam, b_cam, chi_cam] = build_camera_edge_linear_system(...);
   auto [H_ori, b_ori, chi_ori] = build_orientation_linear_system(...);

   H = H_icp + H_cam + H_ori / beta;
   b = b_icp + b_cam + b_ori / beta;

This is the same pattern that ``build_orientation_linear_system`` already
uses (see ``rko_lio/core/lio.cpp``). Each block is independently normalized
in its own builder so there are no scaling surprises.

Visible-point culling
^^^^^^^^^^^^^^^^^^^^^

To keep the inner ICP loop cheap, we collect the visible map points **once
per scan** at the predicted pose, before entering the iteration loop:

- Iterate the Bonxai voxel grid; for each voxel take its representative
  point (the centroid of stored points).
- Project to camera frame; cull points with :math:`p_{c,z} <`
  ``camera_min_point_depth_m`` or with :math:`u` outside the image rect.
- Optionally apply a low-resolution z-buffer at
  :math:`1 / \text{camera\_zbuf\_downsample}` resolution. A point is kept
  only if its depth is within tolerance of the buffer at its pixel.

The resulting list of ``(p_o, voxel_key)`` is reused unchanged across all
Gauss-Newton iterations of the current scan.

Dynamic-point segmentation
--------------------------

The same final-iteration correspondences that build the ICP system also feed
a per-voxel dynamic statistic. ICP converges globally on a single 6-DoF pose,
so "non-converging voxels" is reframed as **voxels whose correspondences end
the ICP loop with high residual or are rejected by the** ``max_correspondance_distance``
gate.

Side-table layout
^^^^^^^^^^^^^^^^^

We keep a side hash table parallel to ``SparseVoxelGrid`` (no change to
``VoxelBlock``):

.. code-block:: cpp

   struct VoxelDynStats {
     float    resid_ema      = 0.f;
     float    dyn_score      = 0.f;
     uint16_t hit_count      = 0;
     uint16_t miss_count     = 0;
     uint32_t last_seen_scan = 0;
   };

   std::unordered_map<Bonxai::CoordT, VoxelDynStats, CoordTHash> voxel_dyn;

This is owned by ``LIO`` and updated only when
``dynamic_segmentation_enabled = true``.

EMA update
^^^^^^^^^^

For each correspondence captured on the final ICP iteration with residual
:math:`r_i` and "accepted" flag :math:`a_i` (true iff
:math:`r_i <` ``max_correspondance_distance``), the matched voxel's stats
are updated as:

.. math::

   \mathrm{resid\_ema} \leftarrow
   (1 - \alpha_d) \cdot \mathrm{resid\_ema} + \alpha_d \cdot r_i

   \mathrm{dyn\_score} \leftarrow
   (1 - \alpha_d) \cdot \mathrm{dyn\_score} +
   \alpha_d \cdot \mathbb{1}\big[r_i > \tau_{\mathrm{dyn}}\;\vee\;\neg a_i\big]

with :math:`\alpha_d =` ``dyn_ema_alpha``. ``hit_count`` is incremented and
``last_seen_scan`` is updated.

Voxels with ``hit_count < dyn_min_hits_to_trust`` are not trusted yet (newly
observed surfaces look "dynamic" until the EMA settles, and the gate handles
that gracefully).

Uses of ``dyn_score``
^^^^^^^^^^^^^^^^^^^^^

In priority order:

1. **Map cleanup**: scan points whose voxel has ``dyn_score >
   dyn_skip_map_score`` are dropped from ``SparseVoxelGrid::Update``. This
   prevents dynamic returns from polluting the map.
2. **ICP weighting (next scan)**: multiply per-point Jacobian and residual
   contributions in ``build_icp_linear_system`` by
   :math:`w_i = \exp(-k \cdot \mathrm{dyn\_score})` with ``k =
   dyn_weight_decay_k``. Self-reinforcing.
3. **Camera mask**: in the visible-point pre-pass for the camera residual,
   drop voxels whose ``dyn_score > dyn_skip_map_score``. Keeps moving cars
   and pedestrians out of the camera alignment.
4. **Publishing**: optionally split ``rko_lio/frame`` into
   ``rko_lio/frame_static`` and ``rko_lio/frame_dynamic``.

Limitations
^^^^^^^^^^^

- **Ego-velocity-matched objects** (adjacent-lane vehicle at the same speed)
  produce small relative residuals and are labeled static. Residual-only
  methods cannot detect them; visibility-based ray-casting
  (Removert / Dynablox / ERASOR) is the path forward and is queued for v2.
- **Miscalibrated extrinsics** inflate ``resid_ema`` globally and can
  produce spurious dynamic labels. The ``hit_count`` gate plus the
  per-voxel EMA mitigate transient noise.
- **Newly observed surfaces** transiently look dynamic. The
  ``dyn_min_hits_to_trust`` gate prevents premature classification.

Map-frame correction propagation
--------------------------------

When camera keyframes are enabled, the corrected pose at the keyframe time
:math:`t_n` is converted into a delta in the odom frame:

.. math::

   \Delta T_n = T^*(t_n) \cdot T_{\mathrm{lio}}(t_n)^{-1}

This delta is published as the ``map -> odom`` TF (REP-105). The
``odom -> base_link`` chain remains exactly the raw LIO output (smooth,
never jumps); ``map -> base_link`` inherits the camera correction via TF
composition. Because the RKO-LIO local map lives in ``odom`` and never
moves, the camera correction does not invalidate any stored map points.

Config keys
-----------

Defaults shown.

Camera:

.. code-block:: yaml

   camera_enabled: false
   camera_weight: 1.0
   camera_max_dt_residual_px: 20.0
   camera_min_visible_points: 200
   camera_min_point_depth_m: 0.5
   camera_use_occlusion_zbuf: true
   camera_zbuf_downsample: 4
   camera_warmup_scans: 5

Dynamic segmentation:

.. code-block:: yaml

   dynamic_segmentation_enabled: false
   dyn_tau_static_m: 0.10
   dyn_tau_dynamic_m: 0.30
   dyn_ema_alpha: 0.20
   dyn_skip_map_score: 0.60
   dyn_weight_decay_k: 4.0
   dyn_min_hits_to_trust: 3

References
----------

- VLOAM / DEMO / LIMO: depth-from-LiDAR for monocular visual odometry.
- KISS-ICP and Kinematic-ICP: ICP backbone that RKO-LIO inherits.
- Removert, Dynablox, ERASOR: visibility-based dynamic map carving
  (queued for v2 of dynamic segmentation).
