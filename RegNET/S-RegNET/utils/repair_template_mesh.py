"""Remove the template white surface's own self-intersections.

Why this exists
---------------
The genus-0 deliverable is scored by self-intersection, not by flip% (see
mesh_flip_probe.py). Of the ~0.53% self-intersecting faces measured on a pushed
mesh, ~0.10% is already present in the TEMPLATE — FreeSurfer's own
OASIS_OAS1_0406_MR1 ?h.white crosses itself in 660 faces / 47 patches before any
warp touches it. That fifth of the total defect is free to remove: it needs no
retraining, only a better template.

What it does
------------
Moves vertices, never faces. V, E and F are untouched, so V-E+F cannot change and
the surface stays combinatorially genus-0 by construction — the only thing being
repaired is the embedding. Intersecting patches are found, dilated by a few rings,
and Laplacian-relaxed with a falloff that pins the outer ring so no crease is
introduced at the patch boundary; then the whole mesh is re-tested. Repeat until
nothing crosses.

Both hemispheres are repaired in one pass over the joined mesh. That is not just
convenience: crossings are detected on the joined mesh, and a per-hemisphere tool
would be blind to an lh-vs-rh crossing. (On 0406 there are none — the flagged
patches sit 12.6 mm clear of the midline — but the check is cheap to keep honest.)

Numpy plus nibabel, nothing else, so it runs anywhere the surfaces do — the rest
of utils/ pulls in torch, scipy and matplotlib, which the mesh boxes need not
have. It therefore carries its own detector rather than importing the probe's;
the two agree exactly on 0406 (660 faces / 0.1007% / 47 patches / largest 267)
from completely different broadphases, which is what validates both.

Measured on OASIS_OAS1_0406_MR1: 660 -> 0 crossing faces in 11 rounds, moving
1,950 verts (0.595%), median 0.16 mm, and 72% of even that motion is tangential.

Usage
-----
    # the normal run — repaired lh/rh.white land in <out_dir>
    python utils/repair_template_mesh.py --template_surf <lh.white> <rh.white> \
                                         --out_dir <dir>

    # let the config locate the template surfaces instead (needs pyyaml)
    python utils/repair_template_mesh.py --config config.yaml

    # measure only, or work straight off a probe PLY with numpy alone
    python utils/repair_template_mesh.py --ply template_..._selfint.ply --detect_only

Feed the result to any of the mesh tools with
    --template_surf <out_dir>/lh.white <out_dir>/rh.white
"""

import argparse
import json
from pathlib import Path

import numpy as np

# =============================================================================
# Detection — same test as mesh_flip_probe.self_intersections, no scipy
# =============================================================================

def seg_hits_tri(P, Q, T, eps=1e-14):
    """Möller-Trumbore: does the segment P->Q pierce triangle T? All (N,3)/(N,3,3)."""
    D, e1, e2 = Q - P, T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    pv = np.cross(D, e2)
    det = (e1 * pv).sum(1)
    ok = np.abs(det) >= eps                       # parallel segment: no crossing
    inv = np.zeros_like(det)
    np.divide(1.0, det, out=inv, where=ok)
    tv = P - T[:, 0]
    qv = np.cross(tv, e1)
    u = (tv * pv).sum(1) * inv                    # barycentric in T
    w = (D * qv).sum(1) * inv
    t = (e2 * qv).sum(1) * inv                    # position along the segment
    return ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t >= 0) & (t <= 1)


def tri_tri_cross(A, B):
    """Do the paired triangles A[i], B[i] cross? Two triangles intersect iff an
    edge of one pierces the other, so six segment tests decide it. Coplanar
    overlap is measure-zero and is not detected."""
    hit = np.zeros(len(A), dtype=bool)
    for X, Y in ((A, B), (B, A)):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            hit |= seg_hits_tri(X[:, a], X[:, b], Y)
    return hit


def cell_pairs(lo, hi, h):
    """Candidate pairs: triangles whose boxes share a grid cell.

    Every triangle is inserted into each cell its box overlaps. A triangle
    several times the typical size then pays for itself in extra cells, instead
    of forcing every other triangle to be queried at ITS radius — the failure
    mode that makes a single-radius neighbour query blow up on a stretched
    surface."""
    c_lo = np.floor(lo / h).astype(np.int64)
    c_hi = np.floor(hi / h).astype(np.int64)
    span = c_hi - c_lo + 1                                   # cells per axis
    cnt = span.prod(1)
    tri = np.repeat(np.arange(len(lo)), cnt)
    k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    nx, ny = np.repeat(span[:, 0], cnt), np.repeat(span[:, 1], cnt)
    off = np.stack([k % nx, (k // nx) % ny, k // (nx * ny)], 1)
    g = np.repeat(c_lo, cnt, axis=0) + off - c_lo.min(0)     # non-negative cell
    dim = c_hi.max(0) - c_lo.min(0) + 1
    ids = (g[:, 2] * dim[1] + g[:, 1]) * dim[0] + g[:, 0]

    order = np.argsort(ids, kind='stable')
    ids_s, tri_s = ids[order], tri[order]
    start = np.flatnonzero(np.r_[True, ids_s[1:] != ids_s[:-1]])
    size = np.diff(np.r_[start, len(ids_s)])
    out = []
    for s in np.unique(size[size > 1]):                      # a few distinct sizes
        blk = tri_s[start[size == s][:, None] + np.arange(s)]
        a, b = np.triu_indices(s, 1)
        out.append(np.stack([blk[:, a].ravel(), blk[:, b].ravel()], 1))
    if not out:
        return np.zeros((0, 2), np.int64)
    p = np.vstack(out)
    p.sort(1)                                                # i<j, then dedupe
    key = p[:, 0] * (len(lo) + 1) + p[:, 1]
    return p[np.unique(key, return_index=True)[1]]


def clusters(faces, sel):
    """(n_clusters, largest) over the SELECTED faces, adjacency = shared vertex.
    Union-find in plain python: the selection is a few hundred faces."""
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        return 0, 0
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for f in idx:                                   # a face joins with its verts
        for node in (('f', int(f)), *(('v', int(v)) for v in faces[f])):
            parent.setdefault(node, node)
        rf = find(('f', int(f)))
        for v in faces[f]:
            rv = find(('v', int(v)))
            if rv != rf:
                parent[rv] = rf
                rf = find(rf)
    lab = {}
    for f in idx:
        lab[find(('f', int(f)))] = lab.get(find(('f', int(f))), 0) + 1
    return len(lab), max(lab.values())


def detect(v, faces, cell=2.0, chunk=2_000_000, label='', quiet=False):
    """Faces of this mesh that cross a non-adjacent face of the same mesh.

    Moving vertices cannot change V-E+F, so a warped mesh is always
    combinatorially genus-0; the failure a warp CAN cause is the surface passing
    through itself, and this is that measurement. Pairs sharing a vertex are
    dropped — neighbours always touch at it."""
    tri = v[faces].astype(np.float64)
    lo, hi = tri.min(1), tri.max(1)
    rad = np.linalg.norm(tri - tri.mean(1)[:, None, :], axis=2).max(1)
    pairs = cell_pairs(lo, hi, cell * float(np.percentile(rad, 99.0)))
    if not quiet:
        print(f"      [self-int {label}] {len(pairs):,} candidates", end='', flush=True)

    hit, n_pairs, n_narrow = np.zeros(len(faces), dtype=bool), 0, 0
    for s in range(0, len(pairs), chunk):
        i, j = pairs[s:s + chunk].T
        m = (lo[i] <= hi[j]).all(1) & (lo[j] <= hi[i]).all(1)             # boxes overlap
        i, j = i[m], j[m]
        m = ~(faces[i][:, :, None] == faces[j][:, None, :]).any((1, 2))   # not neighbours
        i, j = i[m], j[m]
        n_narrow += len(i)
        x = tri_tri_cross(tri[i], tri[j])
        n_pairs += int(x.sum())
        hit[i[x]] = True
        hit[j[x]] = True
    if not quiet:
        print(f" -> {n_narrow:,} narrowphase -> {n_pairs:,} crossings", flush=True)

    n_c, big = clusters(faces, hit)
    return {'si_candidates': int(len(pairs)), 'si_narrowphase': int(n_narrow),
            'si_pairs': int(n_pairs), 'si_faces': int(hit.sum()),
            'si_faces_pct': float(hit.mean() * 100),
            'si_clusters': n_c, 'si_largest': big}, hit


# =============================================================================
# Repair — vertices only
# =============================================================================

def adjacency(faces, n_v):
    """Undirected vertex adjacency as (src, dst, degree)."""
    e = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    e = np.vstack([e, e[:, ::-1]])
    key = np.unique(e[:, 0] * n_v + e[:, 1])
    src, dst = key // n_v, key % n_v
    deg = np.maximum(np.bincount(src, minlength=n_v), 1)
    return src, dst, deg


def umbrella(v, src, dst, deg):
    """mean(neighbours) - v, the discrete Laplacian. bincount, not add.at: same
    scatter-add, ~50x faster at 4M edges."""
    s = np.stack([np.bincount(src, v[dst, c], minlength=len(v)) for c in range(3)], 1)
    return s / deg[:, None] - v


def rings(seed, src, dst, k):
    """Ring index per vertex: 0 on the seeds, r on the r-th ring out, -1 beyond."""
    ring = np.full(len(seed), -1, np.int32)
    ring[seed] = 0
    cur = seed
    for r in range(1, k + 1):
        nb = np.zeros(len(ring), bool)
        nb[dst[cur[src]]] = True
        cur = nb & (ring < 0)
        ring[cur] = r
    return ring


def repair(v0, faces, k=2, lam=0.3, inner=5, max_iter=60, cell=2.0, k_max=8):
    """Relax the intersecting patches until nothing crosses.

    Each round: find the crossing faces, dilate their vertices by k rings, and
    Laplacian-smooth that set with a weight that falls linearly to zero at the
    outer ring — the boundary is pinned, so smoothing a patch cannot leave a
    crease around it. A round that fails to reduce the crossing count widens the
    patch instead of pushing harder, since a persistent interpenetration is
    usually deeper than the current neighbourhood."""
    v = v0.astype(np.float64).copy()
    src, dst, deg = adjacency(faces, len(v))
    hist, prev = [], None
    for it in range(1, max_iter + 1):
        st, hit = detect(v, faces, cell=cell, label=f'iter {it}')
        hist.append(st)
        if st['si_pairs'] == 0:
            print(f"      [repair] clean after {it - 1} round(s)", flush=True)
            break
        if prev is not None and st['si_pairs'] >= prev and k < k_max:
            k += 1
            print(f"      [repair] no progress -> widening to k={k}", flush=True)
        prev = st['si_pairs']

        ring = rings(np.isin(np.arange(len(v)), faces[hit]), src, dst, k)
        w = np.where(ring >= 0, lam * (1.0 - ring / (k + 1.0)), 0.0)[:, None]
        for _ in range(inner):
            v += w * umbrella(v, src, dst, deg)
        d = np.linalg.norm(v - v0, axis=1)
        print(f"      [repair] iter {it}: {st['si_faces']} faces / {st['si_pairs']} pairs"
              f" | {int((w > 0).sum()):,} verts active | disp max {d.max():.3f} "
              f"mean {d[d > 0].mean() if (d > 0).any() else 0:.4f} mm", flush=True)
    else:
        print(f"      [repair] STOPPED at max_iter with "
              f"{hist[-1]['si_pairs']} crossing pair(s) left", flush=True)
    return v, hist


def vertex_normals(v, faces):
    """Area-weighted vertex normals (the cross product is 2*area*n, so summing it
    over incident faces weights by area for free)."""
    tri = v[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.stack([np.bincount(faces.ravel(), np.repeat(fn[:, c], 3), minlength=len(v))
                   for c in range(3)], 1)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)


def displacement_report(v0, v1, faces):
    """What the repair cost, split into the part that matters and the part that
    doesn't: motion along the original normal moves the surface (changes
    anatomy), motion tangent to it only reparameterises the surface and leaves it
    where it was. On 0406 ~72% of the motion is tangential."""
    disp = v1 - v0
    d = np.linalg.norm(disp, axis=1)
    moved = d > 1e-9
    vn = vertex_normals(v0, faces)
    dn = (disp * vn).sum(1)
    dt = np.linalg.norm(disp - dn[:, None] * vn, axis=1)
    e0 = np.linalg.norm(v0[faces[:, [1, 2, 0]]] - v0[faces], axis=2)
    e1 = np.linalg.norm(v1[faces[:, [1, 2, 0]]] - v1[faces], axis=2)
    q = lambda x, p: float(np.percentile(x[moved], p)) if moved.any() else 0.0  # noqa: E731
    return {'verts_moved': int(moved.sum()),
            'verts_moved_pct': float(moved.mean() * 100),
            'disp_max_mm': float(d.max()),
            'disp_mean_moved_mm': float(d[moved].mean()) if moved.any() else 0.0,
            'disp_p50_mm': q(d, 50), 'disp_p95_mm': q(d, 95),
            'verts_over_1mm': int((d > 1.0).sum()), 'verts_over_2mm': int((d > 2.0).sum()),
            'normal_mean_abs_mm': float(np.abs(dn[moved]).mean()) if moved.any() else 0.0,
            'normal_max_abs_mm': float(np.abs(dn).max()),
            'inward_pct': float((dn[moved] < 0).mean() * 100) if moved.any() else 0.0,
            'tangential_share_pct': float(dt[moved].sum() / d[moved].sum() * 100)
            if moved.any() else 0.0,
            'edge_mean_before_mm': float(e0.mean()),
            'edge_mean_after_mm': float(e1.mean())}


# =============================================================================
# I/O
# =============================================================================

def read_ply(path):
    """Verts (world mm) + faces from a binary PLY as written by mesh_flip_probe."""
    with open(path, 'rb') as f:
        head = b''
        while not head.endswith(b'end_header\n'):
            head += f.read(1)
        txt = head.decode('ascii')
        n_v, n_f = (int([l for l in txt.splitlines()
                         if l.startswith(f'element {e}')][0].split()[-1])
                    for e in ('vertex', 'face'))
        vd = np.frombuffer(f.read(n_v * 15), dtype=[('x', '<f4'), ('y', '<f4'),
                                                    ('z', '<f4'), ('r', 'u1'),
                                                    ('g', 'u1'), ('b', 'u1')])
        fd = np.frombuffer(f.read(n_f * 13), dtype=[('n', 'u1'), ('a', '<i4'),
                                                    ('b', '<i4'), ('c', '<i4')])
    verts = np.stack([vd['x'], vd['y'], vd['z']], 1).astype(np.float64)
    faces = np.stack([fd['a'], fd['b'], fd['c']], 1).astype(np.int64)
    return verts, faces


def write_ply(path, verts, faces, color):
    """Binary little-endian PLY with per-VERTEX colour, byte-identical in layout to
    the one mesh_flip_probe writes, so both load into the same viewer and can be
    compared side by side. Vertex colour, not face colour: face colour is the half
    of the spec that online viewers routinely ignore."""
    vd = np.empty(len(verts), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for k, i in (('x', 0), ('y', 1), ('z', 2)):
        vd[k] = verts[:, i]
    for k, i in (('red', 0), ('green', 1), ('blue', 2)):
        vd[k] = color[:, i]
    fd = np.empty(len(faces), dtype=[('n', 'u1'), ('a', '<i4'), ('b', '<i4'), ('c', '<i4')])
    fd['n'] = 3
    for k, i in (('a', 0), ('b', 1), ('c', 2)):
        fd[k] = faces[:, i]
    assert vd.itemsize == 15 and fd.itemsize == 13, 'structured dtype got padded'
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
            "end_header\n")
    with open(path, 'wb') as f:
        f.write(head.encode('ascii'))
        f.write(vd.tobytes())
        f.write(fd.tobytes())
    print(f"[ply] {Path(path).name}  ({Path(path).stat().st_size / 1e6:.1f} MB)", flush=True)


def selfint_colors(faces, hit, n_verts):
    """grey clean, red crossing. Same convention as the probe's PLY, so on a
    repaired mesh this is simply all grey — which is the point of looking."""
    col = np.full((n_verts, 3), 200, np.uint8)
    if hit.any():
        col[np.unique(faces[hit])] = (220, 30, 30)
    return col


def disp_colors(d, cap=None):
    """grey where nothing moved, then blue -> cyan -> yellow -> red with how far it
    did. Lets the eye check whether the deep defect was smoothed into something
    anatomically plausible or just flattened."""
    cap = float(cap or d.max())
    t = np.clip(d / max(cap, 1e-12), 0, 1)
    stops = np.array([[40, 60, 200], [0, 190, 190], [240, 220, 40], [220, 30, 30]], float)
    x = t * (len(stops) - 1)
    i = np.clip(x.astype(int), 0, len(stops) - 2)
    f = (x - i)[:, None]
    col = stops[i] * (1 - f) + stops[i + 1] * f
    col[d <= 1e-9] = 200
    return col.astype(np.uint8), cap


def hemi_split(faces, n_v):
    """n_lh from face connectivity: lh+rh were concatenated with an index offset
    and no face spans the two, so the block boundary is an index no face's
    [min,max) interval covers."""
    d = np.zeros(n_v + 1, np.int64)
    np.add.at(d, faces.min(1), 1)
    np.add.at(d, faces.max(1), -1)
    cover = np.cumsum(d)[:n_v]
    cuts = np.flatnonzero(cover[1:] == 0) + 1
    cuts = cuts[cuts < n_v - 1]
    return int(cuts[0]) + 1 if len(cuts) == 1 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--template_surf', nargs=2, metavar=('LH', 'RH'),
                    help='the template lh/rh white surfaces — needs only nibabel')
    src.add_argument('--config', help='find those surfaces via the config instead '
                                      '(needs pyyaml)')
    src.add_argument('--ply', help='offline input: a probe PLY (world mm), numpy only')
    ap.add_argument('--ref', default=None,
                    help='template .nii.gz, only needed for the optional normalized .npz; '
                         'taken from --config when that is given')
    ap.add_argument('--out_dir', default=None, help='default <template mesh dir>/repaired')
    ap.add_argument('--detect_only', action='store_true', help='measure, change nothing')
    ap.add_argument('--write_ply', action='store_true',
                    help='also write viewer PLYs (~13 MB each): the repaired mesh under '
                         'the probe\'s self-intersection colours, and a displacement map')
    # Defaults come from a sweep on 0406: gentler settings all reach zero crossings,
    # and the gentle ones keep the >2 mm motion inside the one genuinely deep defect
    # instead of spreading it over the whole surface (which lam=0.5, inner=12 did).
    ap.add_argument('--rings', type=int, default=2, help='patch dilation, in vertex rings')
    ap.add_argument('--lam', type=float, default=0.3, help='Laplacian step on the seeds')
    ap.add_argument('--inner', type=int, default=5, help='smoothing steps per round')
    ap.add_argument('--max_iter', type=int, default=60)
    ap.add_argument('--cell', type=float, default=2.0,
                    help='broadphase cell size, in units of the 99th-pct triangle radius')
    args = ap.parse_args()

    ref = meta = None
    n_lh = None
    if args.ply:
        verts, faces = read_ply(args.ply)
        print(f"[in] {args.ply}: {len(verts):,} verts | {len(faces):,} faces (world mm)")
        out = Path(args.out_dir or '.').expanduser()
    else:
        import nibabel as nib                      # only this path needs it
        ref_path = args.ref
        if args.template_surf:
            lh, rh = args.template_surf
        else:
            import yaml                            # and only this one needs pyyaml
            tpl_seg = yaml.safe_load(open(args.config))['data']['template_seg_path']
            ref_path = ref_path or str(tpl_seg).replace('_onehot.npy', '.nii.gz')
            mesh_dir = Path(str(Path(tpl_seg).parent).replace('/scans/', '/meshes/'))
            names = [('lh.white', 'rh.white'), ('lh.white.surf', 'rh.white.surf')]
            pair = next((p for p in ((mesh_dir / a, mesh_dir / b) for a, b in names)
                         if p[0].is_file() and p[1].is_file()), None)
            if pair is None:
                raise FileNotFoundError(f"no {{lh,rh}}.white[.surf] under {mesh_dir}")
            lh, rh = str(pair[0]), str(pair[1])
        if ref_path:
            ref = nib.load(ref_path)
            print(f"[in] ref {ref_path}")
        print(f"[in] {lh}\n[in] {rh}")
        vs, fs, meta = [], [], []
        for p in (lh, rh):
            c, f, m = nib.freesurfer.read_geometry(p, read_metadata=True)
            vs.append(c + m.get('cras', np.zeros(3)))          # surface-RAS -> world
            fs.append(np.asarray(f, np.int64))
            meta.append(m)
        n_lh = len(vs[0])
        verts = np.concatenate(vs)
        faces = np.concatenate([fs[0], fs[1] + n_lh])
        out = Path(args.out_dir) if args.out_dir else Path(lh).parent / 'repaired'

    # The surface path already knows n_lh from the lh vertex count; recover it from
    # face connectivity otherwise, and cross-check when both are available.
    split = hemi_split(faces, len(verts))
    if n_lh is None:
        n_lh = split
    elif split != n_lh:
        print(f"[mesh] WARNING: lh vertex count {n_lh:,} but connectivity says {split}")
    print(f"[mesh] {len(verts):,} verts | {len(faces):,} faces | n_lh = {n_lh}")

    print("[before]")
    st0, hit0 = detect(verts, faces, cell=args.cell, label='template')
    print(f"      {st0['si_faces']} faces ({st0['si_faces_pct']:.4f}%) | "
          f"{st0['si_pairs']} pairs | {st0['si_clusters']} patches | "
          f"largest {st0['si_largest']}")
    if n_lh is not None and st0['si_faces']:
        fv = np.unique(faces[hit0])
        print(f"      lh {int((fv < n_lh).sum())} verts | rh {int((fv >= n_lh).sum())} verts")
        x = verts[fv, 0]
        print(f"      flagged x range [{x.min():.2f}, {x.max():.2f}] mm "
              f"(midline crossings need both signs)")
    if args.detect_only:
        if args.write_ply:
            out.mkdir(parents=True, exist_ok=True)
            write_ply(out / 'input_selfint.ply', verts, faces,
                      selfint_colors(faces, hit0, len(verts)))
        return
    if n_lh is None:
        raise SystemExit("[mesh] no clean lh/rh split in the face connectivity, so the "
                         "repaired hemispheres cannot be written back separately")

    verts1, hist = repair(verts, faces, k=args.rings, lam=args.lam, inner=args.inner,
                          max_iter=args.max_iter, cell=args.cell)

    print("[after]")
    st1, hit1 = detect(verts1, faces, cell=args.cell, label='repaired')
    rep = displacement_report(verts, verts1, faces)
    print(f"      {st0['si_faces']} -> {st1['si_faces']} faces | "
          f"{st0['si_pairs']} -> {st1['si_pairs']} pairs | "
          f"{st0['si_clusters']} -> {st1['si_clusters']} patches")
    print(f"      {rep['verts_moved']:,} verts moved ({rep['verts_moved_pct']:.3f}%) | "
          f"disp p50 {rep['disp_p50_mm']:.3f} mean {rep['disp_mean_moved_mm']:.3f} "
          f"p95 {rep['disp_p95_mm']:.3f} max {rep['disp_max_mm']:.3f} mm | "
          f">1mm {rep['verts_over_1mm']} >2mm {rep['verts_over_2mm']}")
    print(f"      {rep['tangential_share_pct']:.1f}% of the motion is tangential (the "
          f"surface stays put); normal mean {rep['normal_mean_abs_mm']:.3f} "
          f"max {rep['normal_max_abs_mm']:.3f} mm, {rep['inward_pct']:.1f}% inward | "
          f"edge {rep['edge_mean_before_mm']:.4f} -> {rep['edge_mean_after_mm']:.4f} mm")
    assert len(verts1) == len(verts), 'vertex count changed — topology is not preserved'

    out.mkdir(parents=True, exist_ok=True)
    (out / 'repair_report.json').write_text(json.dumps(
        {'before': st0, 'after': st1, 'displacement': rep, 'n_lh': n_lh,
         'iters': hist, 'params': vars(args)}, indent=2))

    if args.write_ply:
        # both in world mm, the same frame the probe's PLY uses, so the before and
        # after files can be loaded together.
        write_ply(out / 'repaired_selfint.ply', verts1, faces,
                  selfint_colors(faces, hit1, len(verts1)))
        d = np.linalg.norm(verts1 - verts, axis=1)
        col, cap = disp_colors(d)
        write_ply(out / 'repaired_disp.ply', verts1, faces, col)
        print(f"[ply] displacement ramp: grey 0, blue -> cyan -> yellow -> red at "
              f"{cap:.3f} mm")
        write_ply(out / 'original_selfint.ply', verts, faces,
                  selfint_colors(faces, hit0, len(verts)))
    # Repaired FreeSurfer surfaces are the deliverable: every downstream tool here
    # already takes --template_surf, so nothing needs a new code path to consume
    # them. Each hemisphere keeps its OWN cras from the file it came from, so the
    # write round-trips exactly.
    if meta is not None:
        import nibabel as nib
        for i, name in enumerate(('lh.white', 'rh.white')):
            keep = (faces.max(1) < n_lh) if i == 0 else (faces.min(1) >= n_lh)
            f = faces[keep] - (0 if i == 0 else n_lh)
            w = verts1[:n_lh] if i == 0 else verts1[n_lh:]
            # volume_info must be carried over, not defaulted: the reader adds cras
            # back to get world coords, so a surface written without it comes back
            # translated by cras. It is [0,0,0] on 0406, which would have hidden
            # this, and self-intersection is translation-invariant either way.
            nib.freesurfer.write_geometry(str(out / name),
                                          w - meta[i].get('cras', np.zeros(3)), f,
                                          volume_info=meta[i])
            print(f"[out] {out / name}  ({len(w):,} verts, {len(f):,} faces)")
        print(f"[use] --template_surf {out / 'lh.white'} {out / 'rh.white'}")

    # The .npz is a convenience only, and it needs the coordinate convention from
    # visualize_mesh (which pulls in torch), so a box without torch still gets the
    # surfaces above rather than an error.
    if ref is not None:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from visualize_mesh import world_to_norm         # noqa: E402
            np.savez(out / 'template_wm_mesh_repaired.npz',
                     verts_norm=world_to_norm(verts1, ref).astype(np.float32),
                     faces=faces.astype(np.int64), n_lh=np.int64(n_lh))
            print(f"[out] {out / 'template_wm_mesh_repaired.npz'}  "
                  f"-> or set data.template_wm_mesh_path to this")
        except ImportError as e:
            print(f"[npz] skipped ({e}); the surfaces above are the deliverable")
    np.savez(out / 'repaired_world.npz',
             verts_world=verts1.astype(np.float32), faces=faces, n_lh=np.int64(n_lh))
    print(f"[out] {out / 'repaired_world.npz'} (world mm, convention-free)")
    print(f"[out] {out / 'repair_report.json'}")


if __name__ == '__main__':
    main()
