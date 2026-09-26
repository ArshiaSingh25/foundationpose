# 0. Beginner's Guide — from zero to this project

**You do not need to know any of this already.** This page builds the
concepts from the ground up, with plain language first and the technical term
after, so the other nine documents become readable.

If you only want the *what*, stop after §3. If you're going to change the code,
continue to the end.

---

## 1. What are we actually trying to find?

The perfume bottle is on a table. We want to know **exactly where it is and how
it is turned**. Those six numbers are called the **6D pose**:

- **3 numbers for position**: how far left/right, up/down, and
  forward/back from the camera.
- **3 numbers for orientation**: which way the bottle is pointing.

Written down, a pose looks like this (from our own logs):

```
t = [  -30.0,     78.8,    182.7] mm   <- position of the bottle's base centre
rpy = [ 24.1,     2.4,   175.1] deg   <- which way it is turned
```

> **Term you'll meet:** *pose*. It just means position + orientation together.
> "6D pose" is a slightly awkward name; there is no 6th dimension. It means 3
> for position and 3 for rotation.

### 1.1 We store a pose as a 4×4 matrix

Numbers grouped into a matrix are easier to apply to a point, so a pose is
stored as a 4×4 array of 16 numbers. The top-left 3×3 holds the rotation, the
first three entries of the last column hold the position, and the rest is
padding.

To "apply" a pose means: take a point described in the bottle's own
coordinates, and the matrix converts it into where that point is, as seen from
the camera. In our code, that is a single matrix multiply.

> **Term:** *homogeneous transform* / *4×4 matrix*. Just the container format
> for "position + rotation". We follow the convention that the **camera's +Z
> axis points straight ahead, +X to the right, +Y down** — so a bottle 30 cm in
> front of the camera has a positive `z` of about 300.

### 1.2 Three ways to write the rotation, and why we print all of them

Rotation is genuinely awkward to write down, so there are several conventions
and people disagree about them:

| Notation | What it means | Watch out for |
|---|---|---|
| **rpy** (roll/pitch/yaw) | tilt side-to-side, tilt front-to-back, spin about the vertical | **Order matters.** "roll-pitch-yaw" and "yaw-pitch-roll" are different answers |
| **rotation vector** | one axis, one angle | Three numbers, but they're coupled |
| **axis in camera frame** | just the direction the bottle's long axis points | Cannot be misread |

Our logs show all three. That's deliberate: if two ever disagree, something is
wrong, and you can spot it immediately. Read the docs with this table open.

### 1.3 An unavoidable ambiguity

The bottle body is a rectangular prism. Spin it 180° about its long axis and it
looks **identical**. So the pose is only ever knowable *up to* that flip — our
logs label it `(mod 180 about long axis)`.

This also explains a measurement in our own test output:

```
dt = 0.07 mm,  dR = 177.97 deg
```

The bottle did **not** jump 178°. It moved 0.07 millimetres. The renderer just
picked the other of two identical answers. Anyone comparing poses across frames
must account for this or they will see phantom jumps.

---

## 2. What the camera gives us

### 2.1 Colour pictures have no distance in them

An ordinary photo is a grid of colours. If a bottle is 1 m away and another is
2 m away, they can occupy the same number of pixels. The photo genuinely does
not contain the information.

So we use an **Intel RealSense depth camera**, which is two cameras in one:

- a normal **colour** sensor giving an RGB grid (red/green/blue per pixel);
- an **infrared stereo pair** that measures the **distance to each pixel**.

The result is two grids of the same size. In our code, together, they are
`(rgb, depth)`.

> **Term:** *RGB* — just the three colour numbers per pixel. *Depth* — the
> distance to that pixel, in **metres**.
>
> **Important:** a depth of `0` means **"no measurement here"**, not "zero
> distance". Shiny glass, black surfaces and distant walls all produce holes.
> This is the single most common source of confusion in this project.

### 2.2 Camera intrinsics: the K matrix

To go from a 3D point to a pixel, you need to know about *this particular
camera* — how wide its field of view is, and where its optical centre sits.
That description is a 3×3 matrix, conventionally called **K** (the
**intrinsics**).

```python
K = [[fx,  0, cx],        # fx, fy: focal length = "how zoomed in" it is
     [ 0, fy, cy],        # cx, cy: optical centre, in pixels
     [ 0,  0,  1]]        # fixed last row
```

Our two cameras differ, and this matters:

| Camera | fx | A 153 mm bottle at 0.4 m spans |
|---|---|---|
| D405 (local laptop) | 392.8 | ~150 px |
| D435i (on the robot) | 605.3 | ~232 px |

> **Term:** *intrinsics*. **Term:** *focal length in pixels* — a bigger number
> means a narrower field of view, so the same object looks bigger.

**The rule that causes real bugs:** depth must be aligned into the **same
camera** as the image whose `K` you use. Both live paths do
`rs.align(rs.stream.color)` and take `K` from the colour stream, so they match.
The D435i's *depth* intrinsics (fx 399) must never be combined with its
*aligned* colour intrinsics (fx 605).

### 2.3 Depth units: three sources, one rule

FoundationPose requires depth in **metres**. We get it from three places in
three formats:

| Source | Stored as | Conversion |
|---|---|---|
| D405 live | `uint16`, sensor units | `× depth_scale` (0.001) |
| D435i streamed | `uint16`, sensor units | `× depth_scale` from the header |
| Recorded PNGs | 16-bit, millimetres | `÷ 1000` |

Get this wrong and the algorithm silently produces nonsense. A related trap: if
you convert metres to `uint16` for streaming, every value truncates to 0 or 1
and the depth is destroyed. (We checked — ours stays in raw sensor units.)

---

## 3. Telling the algorithm which object you mean

A frame often has several things in it. The algorithm needs to be told *which*
one is the perfume bottle. That instruction is a **mask**: a black-and-white
image the same size as the frame, white on the bottle, black everywhere else.

> **Term:** *mask*, *binary mask*, *segmentation*. All the same idea: a
> per-pixel yes/no label. Our logs mention it ~110 times, so it's worth being
> clear that it just means "which pixels belong to the object".

In this project masks come from three places:

1. **A hand-drawn annotation** for the one recorded frame that has one
   (`data/perfume/masks/000000.png`).
2. **A box you drag with the mouse**, refined into a mask by an algorithm
   called **GrabCut**.
3. **A remembered mask** loaded from `init_state.npz`, so you don't have to
   draw anything again.

### 3.1 GrabCut, briefly

You drag a rectangle around the bottle. GrabCut then tries to separate the
pixels inside into "bottle" and "not bottle" by comparing their colours to
what surrounds them.

It is fast, needs no training, and works well — **when the box is tight and
the object is not touching anything else.** When it is not tight, GrabCut
happily returns a confident mask of the wrong object. That exact failure
happened to us; see §6.

### 3.2 About the CAD model

FoundationPose needs to know the bottle's **shape** in advance. We supply it as
a **CAD model** — an exact 3D description, like a very precise physical
photocopy:

- 6068 **vertices** (points in 3D space) and 12128 **faces** (triangles joining
  them).
- True size: **55 × 153 × 30 mm**.

> **Term:** *CAD*, *mesh*, *vertex*, *triangle*. A mesh is just triangles; a
> vertex is a corner. Our mesh is not watertight (has no sealed interior) and
> that is fine — the algorithm only ever looks at the outside.

**Our convention:** the origin is the **centre of the bottle's base**, +Y is
the long axis, so the bottle stands up along +Y. That makes the reported
position directly meaningful: "the middle of the bottle's footprint is here."

### 3.3 Register vs track — the key distinction

FoundationPose does two different jobs, and mixing them up causes most
confusion:

| | `register()` | `track_one()` |
|---|---|---|
| Job | find the bottle from scratch | refine a pose you nearly know |
| Needs a mask? | **Yes** | No |
| Speed | **~2.7 s** | ~30 times per second |
| Called | once, at the start | every frame after |

> **Term:** *registration* — the expensive global search. *tracking* — the cheap
> local refinement.

A fair analogy: registration is searching the whole house for your keys;
tracking is knowing they're on the kitchen table and simply looking there.
Tracking is fast, but it is only fast because it starts from a good guess — and
if that guess is wrong, it will confidently refine the wrong answer. This is
precisely the failure mode this project is built to catch.

---

## 4. The idea that makes this project work

Because the bottle's shape is known exactly, we can **draw** it and compare
that drawing to the real picture. This one idea is used everywhere:

> **Term:** *rendering* — computing what the CAD would look like from the
> camera's point of view. This is what `nvdiffrast` does. It is the same idea
> as tracing a shadow: if the object is in the right place, the drawn outline
> lands exactly on the real outline.

We do this two ways:

### 4.1 Drawing only the outline (silhouette)

Ignore colour and depth; just ask "which pixels does the bottle cover?"

If the pose is right, those pixels match the mask. Compare the two and you get
an **IoU** — *Intersection over Union*, the overlap score between two shapes,
from 0 (disjoint) to 1 (identical).

> **Term:** *silhouette* — the bare outline of a shape. *IoU* — overlap between
> two masks, 0 to 1.

### 4.2 Drawing with depth (this is the important one)

Ask a better question: "**At the pixels where we drew the bottle, is the bottle
actually there?**" Concretely, compare the depth our drawing *predicts* with
the depth the camera *measured*, and ask what fraction agree to within 1 cm.

- Pose correct → we drew the bottle onto the bottle → **high agreement**.
- Pose wrong → we drew the bottle floating in mid-air, or sinking through the
  table → **agreement collapses**.

> **Term:** *inlier fraction* (we call it `fit` in the code) — the fraction of
> the drawing that lands on the real surface. Our logs print it as
> `fit 54%` or `depth agreement 31%`.

**Why this matters so much:** a tracker that has gone wrong keeps returning
confident, well-formed, plausible numbers. Nothing crashes. The failure is
invisible unless you measure it. This measurement needs **no ground truth and
no markers**, so the identical check works on recorded data and on a live feed.

---

## 5. Choosing the thresholds, and why 30%

A threshold is a line you draw: above it, accept the pose; below it, warn.
These only mean something if you know what the *correct* answers score.

We measured all 180 recorded frames, whose answers were independently verified:

| | fit |
|---|---|
| Correct poses | between **43.7%** and **87.3%** (median 58.9%) |
| A deliberately wrong pose | **0%** |

So the gap is `0%` to `43.7%`. We put the line at **30%**, inside the gap.

We considered 50% and **rejected it**: the correct answers spread up to 87% but
start at 43.7%, so a 50% line would have wrongly failed **21 of the 180 good
frames**. The lesson is that a threshold must respect the *worst* good case, not
the average one.

There is a second line, `MIN_IOU = 0.25`. At the correct pose our one annotated
mask scores IoU 0.40, so 0.25 sits below a known-good result.

> **Important caveat, stated in the docs and repeated here:** these numbers were
> measured with the **D405 laptop camera at about 0.19 m**. The robot's D435i
> is used at up to 1.2 m, where "within 1 cm" is a much harder demand. **The
> robot's thresholds are not yet recalibrated.** See
> [`05_POSE_QUALITY.md`](05_POSE_QUALITY.md) §5.2.

---

## 6. Two failure stories worth understanding

### 6.1 Why a check that only warns isn't enough

On a hand-moved bottle, the tracker produced poses that **walked more than a
metre away from the bottle** — a jump of 2171 mm and a rotation of 136.9°
between two consecutive frames.

The fit score *did* drop below the threshold, and the warning *did* print. And
the pose was still written to the log, still drawn on screen, and still used as
the starting guess for the next frame. One bad frame therefore doesn't just get
reported — it gets **fed forward**, and the tracker walks away.

> A speedometer tells you you're speeding. It doesn't slow the car down.

The fix is to **refuse** bad frames rather than report them. This is the
project's main unfinished work; see
[`05_POSE_QUALITY.md`](05_POSE_QUALITY.md) §5.5.

### 6.2 How a mask can be confidently wrong

A box was drawn on the robot's live feed. GrabCut returned a mask of 1601 px,
11% of the box, at a distance of 392 mm. Registration ran and produced a pose
with a depth agreement of 31% and a mask IoU of **0.00** — the drawn bottle and
the real bottle didn't overlap at all.

The arithmetic identified the culprit: a 153 mm bottle at 392 mm spans
**236 × 85 px**, which cannot fit inside the ~121 px box that was drawn. At
1.25 m it spans 74 × 27 px, which fits easily. So GrabCut had segmented
**something small and near — almost certainly the person's hand** — and the
algorithm had dutifully solved the pose of the hand.

The fix was to use the one thing we *do* know: the bottle's size is known from
the CAD, so a blob far smaller than the bottle should be *at that distance* is
not the bottle, and registration is refused before it starts. The check in
`05_POSE_QUALITY.md` §5.6 does exactly that.

---

## 7. Why the computer parts are split across two machines

The project runs on a laptop with an NVIDIA RTX 5080 (a GPU — a chip built for
doing thousands of arithmetic operations in parallel, which is what neural
networks need). The algorithm genuinely requires a fast GPU: **CUDA** is
NVIDIA's software that lets ordinary Python programs use that GPU, and
`nvdiffrast` is the specific piece that draws our CAD.

The robot has its own processor, but it is a **different architecture** (ARM /
`aarch64`, versus the laptop's x86-64). Making the same GPU software work there
would be a substantial project of its own.

So the work is split:

- **On the robot:** be a camera. Open the depth camera, grab frames, compress
  them, send them over the network. That's all.
- **On the laptop:** everything else — the mask, the pose, the checks, the
  drawing.

It costs about 3.3 MB/s of network traffic, which is trivial. The robot doesn't
even need this project's Python code.

The compressed stream (see [`07_G1_SETUP.md`](07_G1_SETUP.md) §7.6) uses two
different compressors on purpose:

- Colour → **JPEG**. Slightly lossy, but photographs survive it, and it is
  small.
- Depth → **zlib**. Lossless, because a lost or altered depth value is a wrong
  distance, and the whole point is measuring distance.

---

## 8. Where to go next

| If you want to… | Go to |
|---|---|
| Understand the intent and the design choices | [`01_OVERVIEW.md`](01_OVERVIEW.md) |
| Understand how the pieces connect | [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) |
| Know what each file does | [`03_FILES.md`](03_FILES.md) |
| Change how FoundationPose is used | [`04_FOUNDATIONPOSE.md`](04_FOUNDATIONPOSE.md) |
| Change a threshold or the quality check | [`05_POSE_QUALITY.md`](05_POSE_QUALITY.md) |
| Change the automatic start-up behaviour | [`06_AUTO_INIT.md`](06_AUTO_INIT.md) |
| Work with the robot or the camera | [`07_G1_SETUP.md`](07_G1_SETUP.md) |
| Just run something | [`08_RUNBOOK.md`](08_RUNBOOK.md) |
| Understand why things are the way they are | [`09_EXPERIMENT_LOG.md`](09_EXPERIMENT_LOG.md) |
| Look up a word | [`10_GLOSSARY.md`](10_GLOSSARY.md) |

### Sanity check: can you answer these?

1. Why does the pose have six numbers, and why is one direction ambiguous?
2. What does a depth value of `0` mean?
3. Why must depth be aligned to the colour camera before using it with `K`?
4. What is the difference between `register()` and `track_one()`, and which one
   needs a mask?
5. Why does the fit score collapse when the tracker goes wrong?
6. Why is the threshold 30% and not 50%?
7. Why is the GPU work done on the laptop and not the robot?
8. Why does a mask IoU of 0.00 mean the mask is wrong?

If any of these are unclear, re-read the matching section above — each answer is
a short paragraph somewhere in §1–§7.
