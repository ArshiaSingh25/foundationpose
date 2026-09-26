# 7. G1 Robot Setup

## 7.1 The machine

| Property | Value |
|---|---|
| Hostname | `unitree-g1-nx` |
| Address | `192.168.1.39` |
| Login | `unitree` / password `123` |
| Architecture | **aarch64**, Jetson Orin NX |
| Local machine | `192.168.1.22`, x86-64, RTX 5080 (`sm_120`, 15 GiB) |
| Python on robot | `python3` with `pyrealsense2` and librealsense 2.56 |

Helpers on the local machine: `/tmp/opencode/askpass.sh` (mode 700) and
`/tmp/opencode/g1`.

```bash
DISPLAY= SSH_ASKPASS=/tmp/opencode/askpass.sh SSH_ASKPASS_REQUIRE=force \
  ssh -o StrictHostKeyChecking=no unitree@192.168.1.39 '<cmd>'
```

## 7.2 Why the pose is not computed on the robot

The local FoundationPose install is x86-64 with a CUDA `nvdiffrast` extension
built for `sm_120`. The G1 is aarch64. Making it run there means an ARM
toolchain, an ARM PyTorch, and an ARM nvdiffrast build — a separate project.

So the split is: **the robot acquires, the local machine infers.** The robot
needs only `pyrealsense2`, `opencv` and `numpy`; it never imports
`perfume_common` and does not need the project's environment.

Cost: ~3.3 MB/s at 28 fps, which is nothing for gigabit Ethernet.

## 7.3 The camera

| Property | Value |
|---|---|
| Device | Intel RealSense **D435i** |
| Serial | `243322071868` |
| USB | `Bus 001 Device 013: ID 8086:0b3a` |
| Resolution | 640 × 480 @ 30 |
| Depth scale | `0.0010000000474974513` m/unit |
| Colour intrinsics | `fx=605.34, fy=604.82, cx=323.5, cy=242.0` |
| Depth intrinsics | `fx=399.26, fy=399.26, …` |

Note the D435i's colour focal length (~605 px) is much longer than the D405's
(~393 px), so the bottle is roughly 1.5× larger in the frame at the same
distance. Anything distance-dependent must be recomputed from `K` — which
`expected_bbox_px()` does.

**The D435i enumerates as USB 2.0 "high-speed", not SuperSpeed.** At 3.3 MB/s
bandwidth is not the constraint, so the drops are a link-quality problem
(cable, contact, port, or power), not throughput.

## 7.4 Port 8765 is occupied — use 8766

Port **8765** is held by `/home/unitree/Downloads/Bidyut_Dashboard/go2_unified_service.py`.
**Do not kill it.** The streamer uses **8766** for this reason.

## 7.5 Files on the robot

| Path | What |
|---|---|
| `/home/unitree/g1_stream.py` | The TCP streamer. Source of truth is `/tmp/opencode/g1_stream.py`; verify with `md5sum` on both sides before assuming they match. |
| `/home/unitree/g1_burst.py` | Capture aligned RGB+depth+K to `.npz`. |
| `/home/unitree/g1_stream.log` | Streamer log. |
| `/home/unitree/g1_soak.py`, `rs_probe.py`, `rs_ladder.py` | USB/RealSense diagnostics. |
| `/home/unitree/g1_burst.npz`, `g1_burst2.npz` | Captured frames. |

Deploy with `scp`, then compare checksums:

```bash
md5sum /tmp/opencode/g1_stream.py
# remote: md5sum /home/unitree/g1_stream.py
```

## 7.6 The streaming protocol

One client at a time. A 4-byte big-endian length prefix, then:

```
payload = [color_len u32][depth_len u32][frame_idx u32]
          [JPEG colour bytes]
          [zlib-compressed raw uint16 depth, row-major HxW]
```

Immediately after the client connects, the server sends a **header** the same
way:

```json
{"w":640,"h":480,"depth_scale":0.0010000000474974513,
 "K":[605.34,604.82,323.5,242.0],
 "name":"Intel RealSense D435I","serial":"243322071868"}
```

**Depth is sent raw, in the sensor's own units** (`np.asanyarray(dfr.get_data())`,
never converted), and the client applies `depth_scale`. Converting to metres on
the robot and casting to `uint16` would truncate every value to 0 or 1 — this was
checked and is correct as written.

Colour is `bgr8` (native). The viewer converts BGR→RGB for drawing, because the
drawing helpers assume RGB (`is_input_rgb=True`), then back to BGR for
`cv2.imshow`. Skipping this swaps red and blue in the preview; it does not
affect the pose.

## 7.7 Resilience behaviour, and two bugs found the hard way

The streamer is written to survive a camera that keeps dropping, because it
does. On a stream error it stops the pipeline, **keeps the socket open**, and
retries `pipe.start()` until the camera re-enumerates.

Two bugs came out of that design:

1. **The header was only ever sent once.** The original code guarded it with
   `if not outage:`, so any client connecting *after* the first outage read raw
   frame bytes as a header and died with
   `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8f`. Symptom: a tight
   reconnect loop, with the streamer log filling with
   `client connected from (...)` followed by a handful of frames each time.
   Fixed by tracking `sent_hdr` **per connection** instead of globally.

2. **The client gave up during a camera outage.** The viewer had a 30 s socket
   timeout, but the streamer deliberately holds the socket open while it waits
   for the camera — so a perfectly healthy session was torn down. The client
   now distinguishes the two cases: `socket.timeout` → print a notice and keep
   waiting (180 s tolerance); `ConnectionError`/`OSError` → genuinely reconnect.

## 7.8 Operating the streamer

```bash
# remote
cd ~ && nohup setsid python3 -u g1_stream.py 8766 80 > g1_stream.log 2>&1 < /dev/null &

# check
pgrep -af g1_stream
ss -ltn | grep 8766
tail -5 g1_stream.log
```

**Do not stop it with `pkill -f "g1_stream.py"` from inside an SSH one-liner.**
`pkill -f` matches the pattern against full command lines — including the
`bash -c` wrapper that is running your own command — so it kills its own shell
and the rest of the command never executes. This cost one confusing round-trip.
Kill by PID, or use a pattern that cannot match the shell.

## 7.9 Observed camera stability

| Event | Detail |
|---|---|
| Behind a Realtek RTS5411 USB 2 hub | Repeated disconnect/re-enumeration |
| Moved to a direct port | Soak test: **300/300 frames, zero timeouts, 26.9 fps, 98.3% valid depth** |
| Live streaming | 2400 frames at 28–29 fps, 3.3 MB/s, then disconnect at ~83 s |
| Later attempts | Drops at 1082 frames, and a period of `Frame didn't arrive within 2000` |
| Camera present during those drops? | **Yes** — `lsusb` still showed `8086:0b3a` |

So the camera survives; the *stream* is what breaks. Contributing factor
observed during one investigation: **load average 52.7** on the Jetson with a
python3 process at 91.9% CPU. The streamer itself is a plausible candidate for
that load. Reducing the Jetson's load is worth trying before blaming the cable.

**Recommended hardware fix, still outstanding:** move the D435i to a genuine
USB 3.x port or a SuperSpeed hub so it enumerates at 5000 Mbps, and reseat the
cable.

## 7.10 Scene guidance for the G1

The D435i's minimum useful range and the metric's tolerance both favour a
nearby subject:

- **0.3–0.5 m** is the working range.
- At the 1.7 m median seen in one capture, a 153 mm bottle is only ~55 px wide
  and 20 px in its narrow dimension — too small for reliable GrabCut
  segmentation or stereo depth.
- One capture with no bottle in view: nearest depth 1.29 m, median 1.76 m,
  **0.00%** of pixels within 0.6 m. In that state no registration is possible,
  which is exactly what happened on the last G1 attempt.
- Keep the whole bottle inside the box, with nothing else inside it and
  nothing in front of it. GrabCut will otherwise segment the nearer object —
  a hand, typically — and the projected-size guard will (correctly) refuse.
