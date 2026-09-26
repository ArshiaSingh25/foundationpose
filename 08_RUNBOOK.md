# 8. Runbook

## 8.1 Environment preamble

Required for **every** FoundationPose invocation. `CUDA_HOME` and
`PYTHONPATH` are both mandatory — without them nvdiffrast fails to load its
compiled plugin.

```bash
FP=/home/priyanshi/miniconda3/envs/foundationpose
cd /home/priyanshi/Downloads/perfume_tracking
export CUDA_HOME=$FP/cuda_home
export PATH=$FP/cuda_home/bin:$PATH
export LD_LIBRARY_PATH=$FP/cuda_home/lib64
export PYTHONPATH=/home/priyanshi/.cache/torch_extensions/py311_cu128/nvdiffrast_plugin
```

For anything with an OpenCV window (this is a headless box, so the display must
be set explicitly):

```bash
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
```

## 8.2 Offline — the verified path

```bash
$FP/bin/python -u run_perfume.py --no_show --debug 2
```

- ~180 frames; `register()` on frame 0, `track_one()` after.
- Output: `debug_perfume/poses.json` (all poses) and
  `debug_perfume/ob_in_cam/<frame>.txt` (4×4 per frame).
- Add `--no_show` to run headless; omit it to get a live overlay.

### Verify it

```bash
$FP/bin/python -u check_iou.py                  # silhouette IoU vs the mask
$FP/bin/python -u check_tracking.py             # per-frame depth agreement
$FP/bin/python -u verify_pose.py                # reprojection sanity check
```

Expected: 180/180 poses, IoU ~0.40, mean agreement 81.2%, min 67.9%, none
below 60%.

## 8.3 Live — local D405

```bash
$FP/bin/python -u live_perfume.py
```

Keys: `b`/Space draw a box and register · `r` release · `s` snapshot ·
`q`/ESC quit. Poses append to `live_poses.jsonl`.

The local D405 is currently **not running** — it was stopped when work moved to
the G1.

## 8.4 Live — G1 robot

**On the robot:**

```bash
DISPLAY= SSH_ASKPASS=/tmp/opencode/askpass.sh SSH_ASKPASS_REQUIRE=force \
  ssh -o StrictHostKeyChecking=no unitree@192.168.1.39 \
  'cd ~ && nohup setsid python3 -u g1_stream.py 8766 80 > g1_stream.log 2>&1 < /dev/null &'
```

**Locally:**

```bash
$FP/bin/python -u /tmp/opencode/live_g1.py --host 192.168.1.39 --port 8766 \
  > /tmp/opencode/live_g1.log 2>&1 < /dev/null &

tail -f /tmp/opencode/live_g1.log
```

Flags: `--init_state PATH` (default `init_state.npz`), `--no_auto_init`,
`--out` (default `live_g1_poses.jsonl`), `--mesh_file`, `--est_refine_iter`,
`--track_refine_iter`.

**First run of a new scene:** place the bottle at 0.3–0.5 m, press `b`, draw a
box that hugs the whole bottle with nothing else inside it, press Enter. That
registration is saved. **Every later run auto-starts with no input.**

## 8.5 Offline analysis of a G1 capture

```bash
# on the robot
python3 /home/unitree/g1_burst.py 30 /home/unitree/g1_burst2.npz
# copy it down, then
$FP/bin/python -u /tmp/opencode/pose_on_g1.py
```

Useful because it removes the GUI from the loop: capture once, then iterate on
masking and registration without asking the user to redraw anything.

## 8.6 Tests

All in `/tmp/opencode/` (ephemeral — copy them into the project if they matter).

| Test | Checks |
|---|---|
| `test_e2e_init.py` | Register + track on recorded data; prints per-frame `dt` (mm) and `dR` (deg) |
| `test_cache.py` | Cached auto-init: reuse / re-register / refuse |
| `test_thresholds.py` | `MIN_FIT` calibration over the 180 frames |
| `test_gate.py` | Gate accepts a good pose, rejects a shifted one |
| `test_symmetry.py` | Measures the real mesh symmetry |
| `diag_auto.py` | Why depth-only auto-detect fails |
| `test_depth_guard.py` | The CAD projected-size guard |

`test_e2e_init.py` is the regression test to run after touching anything in the
mask or registration path. Baseline: `dt = 0.07 mm`, `dR = 177.97 deg` (the
symmetry, not an error — see [`04_FOUNDATIONPOSE.md`](04_FOUNDATIONPOSE.md) §4.3).

## 8.7 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'Utils'` | `Utils` imported before `perfume_common` | Import `perfume_common` first |
| `ModuleNotFoundError: No module named 'nvdiffrast_plugin'` | `PYTHONPATH` not set | Use the §8.1 preamble |
| `OSError: CUDA_HOME environment variable is not set` | `CUDA_HOME` not set | Use the §8.1 preamble |
| Viewer exits immediately after launching | `DISPLAY` unset | `export DISPLAY=:1 XAUTHORITY=...` |
| `UnicodeDecodeError` in a reconnect loop | Client connected after a camera outage and got no header | Fixed in the current streamer; confirm the deployed `g1_stream.py` matches the local one via `md5sum` |
| `Address already in use` on 8765 | Go2 service owns it | Use 8766 |
| `Frame didn't arrive within 2000` repeatedly | Camera dropped | Leave the client connected; it waits 180 s. Check `lsusb`; reseat the cable / move to USB 3.x |
| `that blob is only WxH px ... GrabCut caught the wrong thing` | Box contained a nearer object (usually a hand) | Draw a tighter box around the bottle only |
| `object is X m away` | Too far | Move to 0.3–0.5 m (G1) or 0.15–0.30 m (D405) |
| `no depth on the object` | Glass/shiny surface, or too far | Move closer; a matte backdrop helps |
| `WARNING: the pose no longer fits the frame` | Tracker diverged | `r` then `b`. **This does not stop the divergence** — see §8.8 |
| `pkill` killed the SSH session | `pkill -f` matched its own shell | Kill by PID |

## 8.8 Standing caveat

**The live tracker has no motion gate.** A diverged pose is reported but still
written to the log, still drawn, and still used to seed the next frame. On
hand-moved sequences the trajectory walks metres away from the bottle. Any
conclusion drawn from `live_poses*.jsonl` must account for this. Fix design and
the unresolved policy question are in
[`05_POSE_QUALITY.md`](05_POSE_QUALITY.md) §5.5.

## 8.9 Housekeeping warnings

- `/tmp/opencode/` is **not** persistent. `live_g1.py`, `g1_stream.py` and every
  test script live there. Anything needed to reproduce the G1 setup belongs in
  the project.
- `init_state.npz` **does not exist yet** — no G1 registration has ever been
  accepted, so auto-init is currently inert on that path.
- `live_poses.jsonl` (331) and `live_poses_run2.jsonl` (4416) both contain
  diverged garbage. They are kept as evidence for §5.5, not as results.
- `debug_g1_live/` and `live_snaps_g1/` are empty.
