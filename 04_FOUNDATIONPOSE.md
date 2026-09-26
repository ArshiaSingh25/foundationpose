# 4. FoundationPose Integration

## 4.1 What FoundationPose actually does here

Two methods are used, and the distinction drives the whole design.

### `est.register(K, rgb, depth, ob_mask, iteration=5)`

The **only** method that needs an object mask, and the only expensive one. It
synthesises renderings of the CAD from a large set of candidate rotations,
matches each against the observed depth silhouette, clusters the survivors, and
refines the best ones with the learned `PoseRefinePredictor`. The returned
`detect_upsample`/clustering size is visible in the logs as
`num original candidates = 252` → `num of pose after clustering: 252`.

Cost: **~2.7 s** on the RTX 5080 (measured, printed as
`registered in 2.70 s`).

Called from: `run_perfume.py` (frame 0), `live_perfume.py` `do_select()`,
`auto_init.try_cached_init()` re-registration.

### `est.track_one(rgb, depth, K, iteration=2)`

Per-frame refinement seeded from the previous pose. **No mask.** Uses the
`networks::TrackPose` bundle with `trans_rep='tracknet'`. This is what makes
live tracking real-time.

Cost: roughly 20–30 Hz on the RTX 5080 (the viewer prints an EMA of `Hz`).

Called from: `run_perfume.py` (frames 1..N), both live loops.

## 4.2 Inputs FoundationPose requires, and how each is met

| Requirement | Source | Note |
|---|---|---|
| `model_pts` | `mesh.vertices` | float32 point cloud from trimesh |
| `model_normals` | `mesh.vertex_normals` | needs consistent winding |
| `mesh` | the trimesh object | used for rendering in scoring |
| `K` | colour-stream intrinsics | **must** match the frame depth was aligned to |
| `rgb` | `uint8` HxWx3 | |
| `depth` | `float32` **metres**, 0 = invalid | D405: `uint16 × depth_scale`. G1: streamed `uint16 × depth_scale` |
| `ob_mask` | GrabCut, or the annotated mask | register only |

**Depth units are the most common silent failure.** FoundationPose expects
metres. The D405 path does `asanyarray(...).astype(np.float32) * depth_scale`;
the G1 path receives raw `uint16` and applies the header's `depth_scale`. The
recorded PNGs are in millimetres and `PerfumerReader` divides by `1e3`. Three
different sources, one convention: metres.

**The colour-visuals crash.** If the mesh carries `TextureVisuals` with a `None`
image, `make_mesh_tensors` segfaults on the image. `load_bottle_mesh` replaces
the visuals with a flat `ColorVisuals` grey to prevent this.

**`nvdiffrast` needs a compiled plugin.** It JIT-builds into
`~/.cache/torch_extensions/py311_cu128/nvdiffrast_plugin`, which must be on
`PYTHONPATH` or every run dies with
`ModuleNotFoundError: No module named 'nvdiffrast_plugin'`. It also needs
`CUDA_HOME` set. Both are in the standard environment preamble.

## 4.3 The CAD and its coordinate convention

`model/perfume_bottle_base.obj`:

- extents **54.98 × 153.0 × 30.0 mm**
- bounds X ∈ [−27.49, 27.49], **Y ∈ [0, 153]**, Z ∈ [−15, 15] mm
- 6068 vertices, 12128 faces, not watertight

The origin is the **centre of the base**, +Y is the long axis, the bottle
"stands up" along +Y. So `translation_mm` from `format_pose` is directly the
base-centre position in the camera frame.

Because the body is a rectangular prism, **the pose is only observable up to a
180° spin about +Y.** `format_pose` therefore says `(mod 180 about long axis)`
and additionally reports `long_axis_cam` — the bottle's own +Y axis expressed
in camera coordinates. Unlike Euler angles, that is unambiguous.

Consequence seen in the logs: a tracked pose can jump by ~178° between frames
(`dR = 177.97 deg` in `test_e2e_init.py`) while the *rendered silhouette is
identical* and the tracked translation moves only 0.07 mm. This is the symmetry
doing its job, **not** a tracking failure. Anything comparing poses across
frames must canonicalise about the long axis first.

## 4.4 The rotation grid

`reset_object()` logs `self.diameter: 0.156` (the 153 mm long axis, rounded) and
`vox_size: 0.0078` (diameter/20). These are derived from the mesh and confirm
the CAD is being interpreted at true metric scale — if the mesh were in
millimetres, `diameter` would read 156.

## 4.5 Rendering for scoring

`score_pose` and `check_iou` both render the CAD with
`nvdiffrast_render(K, H, W, ob_in_cams=..., glctx=est.glctx,
mesh_tensors=est.mesh_tensors, mesh=mesh)` and reuse the estimator's own
`glctx` and `mesh_tensors`. Returns `color`, `dep` (rendered depth) and normals;
`color[0].sum(-1) > 0` is the silhouette test.

Consequence: scoring is not free. It costs a rasterisation pass per call, which
is why the live loops score **every 5th frame** rather than every frame. The
colour-vs-depth agreement is what the gate is built on, so this is a real
tradeoff, currently resolved in favour of frame rate.

## 4.6 Determinism

`set_seed(0)` is called in both entry points and `set_logging_format()` is
applied, so runs are reproducible and logs are readable.
