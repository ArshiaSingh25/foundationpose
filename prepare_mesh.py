# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to any modifications thereto.  Any use,
# reproduction, disclosure or distribution of this software and any
# modifications thereto is strictly prohibited.

"""
Prepare the perfume bottle CAD for FoundationPose.

Writes a metric, base-centered copy of the source mesh so that the pose
returned by FoundationPose refers to the center of the bottle's base
(origin at the bottom face, +Y up), which is the most interpretable
convention when tracking a standing object.

The transform applied is rigid (rotation-free translation), so it does not
change the geometry or FoundationPose's accuracy in any way -- it only
changes which point of the object the reported pose refers to.
"""

import argparse
import os

import numpy as np
import trimesh


def main():
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_in', type=str,
                      default=f'{code_dir}/model/perfume_bottle.obj')
  parser.add_argument('--mesh_out', type=str,
                      default=f'{code_dir}/model/perfume_bottle_base.obj')
  parser.add_argument('--up_axis', type=str, default='y',
                      help='axis the bottle stands along (y for this CAD)')
  args = parser.parse_args()

  mesh = trimesh.load(args.mesh_in, force='mesh', process=False)
  mesh.merge_vertices()

  if mesh.is_watertight:
    print('mesh is watertight')
  else:
    # Report the size of any open boundary so it can be judged negligible.
    from collections import Counter
    import networkx as nx
    e = np.sort(mesh.edges_unique, axis=1)
    f = np.sort(mesh.faces, axis=1)
    ef = Counter(map(tuple, np.sort(
      np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]).reshape(-1, 2), axis=1)))
    boundary = [ed for ed, c in ef.items() if c == 1]
    g = nx.Graph()
    g.add_edges_from(boundary)
    loops = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    print(f'mesh not watertight: {len(boundary)} boundary edges in '
          f'{len(loops)} loop(s) {loops[:5]}')

  up = {'x': 0, 'y': 1, 'z': 2}[args.up_axis.lower()]
  other = [i for i in range(3) if i != up]
  lo, hi = mesh.bounds

  # Translate so the origin is the center of the object on the bottom face
  # along the up axis, and centered on the two remaining axes.
  shift = np.zeros(3)
  shift[other] = (lo[other] + hi[other]) / 2
  shift[up] = lo[up]

  out = mesh.copy()
  out.apply_translation(-shift)

  out.export(args.mesh_out)

  print(f'source   : extents={np.round(mesh.extents, 4)} m  '
        f'bounds={np.round(mesh.bounds, 4).tolist()}')
  print(f'shift    : -{np.round(shift, 5).tolist()} m')
  print(f'written  : {args.mesh_out}')
  print(f'extents  : {np.round(out.extents, 4)} m '
        f'= {np.round(out.extents * 1000, 1)} mm')
  print(f'bounds   : {np.round(out.bounds, 5).tolist()}')
  print(f'verts    : {len(out.vertices)}  faces: {len(out.faces)}')


if __name__ == '__main__':
  main()
