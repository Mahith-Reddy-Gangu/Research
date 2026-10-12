"""GT-free clean-up of a pushed white mesh (the "de4" chain). Faces never change.

  1. targets    two sub-voxel signed distances (mm, negative inside, 0.5 mm grid)
                per hemisphere, from the SynthSeg white mask split by nearest
                left / right SynthSeg label:
                  g05   Gaussian (0.5 voxel) of the mask, cubic to 0.5 mm, level 0.5
                  ivb25 interval-constrained biharmonic smoothing of the mask SDF:
                        every 1 mm voxel centre keeps its side, boundary centres
                        stay within 0.25 mm of their binary value
  2. demons     log-domain diffeomorphic demons from the mesh's own SDF to g05,
                one field for both hemispheres (so they cannot cross), 0.5 mm
                grid; every vertex is moved through the map
  3. backtrack  on faces the demons move made cross, scale that move back toward
                the start mesh, then repair whatever still crosses
  4. snap       to ivb25 (40 steps) -> tangential fairing x5 -> snap (20 steps),
                medial-wall vertices get no snap force; every step is
                collision-guarded (crossing vertices go back, their step halves)
  5. repair     a last repair pass if anything still crosses

Inputs only: the start mesh (lh + rh in one .surf, lh first, 163,842 vertices
each, as utils/push_optimized_mesh.py writes it: world - cras, cras = the world
point of the volume centre), the SynthSeg white mask (class 3 of the one-hot
input) and the SynthSeg label volume (hemisphere split; its affine is the
world frame). No GT is read. Run from S-RegNET (needs torch + nibabel + skimage):

    python utils/mesh_cleanup.py --surf <s>_symM75_deformed.white.surf \\
        --input_seg <scans>/<s>/synthseg_white_onehot_v1.npy \\
        --synthseg <synthseg>/<s>/norm_synthseg.nii.gz \\
        --cortex_labels_dir <fsaverage_labels> --output <out>.surf --device cuda:0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from nibabel.affines import apply_affine
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from git_provenance import write_git_sha                                 # noqa: E402
import repair_template_mesh as R                                        # noqa: E402
from push_optimized_mesh import load_cortex_mask                        # noqa: E402

WM = 3                                  # white-surface-interior channel of the one-hot
N_LH = 163842                           # template = fsaverage ico7 per hemisphere
LEFT = (2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 30, 31)       # SynthSeg labels
RIGHT = (41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63)
PAD, BAND, CLIP = 8, 4.0, 8.0           # target crop (voxels), exact-distance band, clip (mm)
SNAP = dict(step=0.25, lam_t=0.1, blend=0.5, cap=3.0, min_gain=0.1, check_every=5)


# =============================================================================
# Frames and IO (world = scanner RAS mm of the label volume)
# =============================================================================

def vox_to_world(ijk, affine):
    return np.asarray(ijk, float) @ affine[:3, :3].T + affine[:3, 3]


def world_to_vox(pts, affine):
    inv = np.linalg.inv(affine)
    return np.asarray(pts, float) @ inv[:3, :3].T + inv[:3, 3]


def as_saved(v, cras):
    """The float32 round trip of writing and re-reading a .surf; the chain was
    built with a save after every step, so every step starts from this."""
    return (v - cras).astype(np.float32).astype(np.float64) + cras


def signed_distance(mask):
    """Voxel-centre signed distance (voxels), negative inside, ~0 between an
    inside and an outside centre."""
    return np.where(mask, -(ndimage.distance_transform_edt(mask) - 0.5),
                    ndimage.distance_transform_edt(~mask) - 0.5).astype(np.float32)


def hemi_masks(mask, labels):
    """(lh, rh) by nearest left- vs right-hemisphere SynthSeg label."""
    dl = ndimage.distance_transform_edt(~np.isin(labels, LEFT))
    dr = ndimage.distance_transform_edt(~np.isin(labels, RIGHT))
    lh = mask & (dl <= dr)
    return lh, mask & ~lh


# =============================================================================
# 1. Targets
# =============================================================================

def upsample(vol, order):
    """1 mm grid -> 0.5 mm grid whose even indices are the 1 mm centres."""
    g = np.broadcast_arrays(*np.meshgrid(*[np.arange(2 * m - 1) / 2.0 for m in vol.shape],
                                         indexing='ij', sparse=True))
    return ndimage.map_coordinates(vol.astype(np.float32), g, order=order, mode='nearest')


def gauss_field(h, sigma=0.5, iso=0.5):
    """g05: Gaussian of the mask, cubic to 0.5 mm, zero level at `iso`."""
    return upsample(iso - ndimage.gaussian_filter(h.astype(np.float32), sigma), 3)


def ivb_field(h, dev, w=0.25, d=0.05, steps=2000, t=0.012):
    """ivb25: `steps` x [f -= t lap(lap f)] from the upsampled mask SDF, after
    each step clamped to per-point bounds: every 1 mm centre keeps its side
    (inside <= -d, outside >= d) and boundary-layer centres (|sd| = 0.5) stay
    within w of their binary value."""
    sd = signed_distance(h)
    n = [2 * m - 1 for m in h.shape]
    lo, hi = np.full(n, -1e9, np.float32), np.full(n, 1e9, np.float32)
    l1, h1 = np.where(h, -1e9, d).astype(np.float32), np.where(h, -d, 1e9).astype(np.float32)
    b = np.abs(sd) <= 0.5
    l1[b], h1[b] = np.maximum(l1[b], sd[b] - w), np.minimum(h1[b], sd[b] + w)
    lo[::2, ::2, ::2], hi[::2, ::2, ::2] = l1, h1
    k = torch.zeros(1, 1, 3, 3, 3, device=dev)
    k[0, 0, 1, 1, 1] = -6
    for a in [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
        k[(0, 0) + a] = 1
    lap = lambda x: F.conv3d(F.pad(x, (1,) * 6, mode='replicate'), k)
    T = lambda a: torch.from_numpy(a)[None, None].to(dev)
    x, lo, hi = T(upsample(sd, 1)), T(lo), T(hi)
    with torch.no_grad():
        for _ in range(steps):
            x -= t * lap(lap(x))
            x = torch.maximum(torch.minimum(x, hi), lo)
    return x[0, 0].cpu().numpy()


def redistance(f):
    """Field f on the 0.5 mm grid (zero level = target) -> SDF in mm: distance
    to its marching-cubes surface (vertices + face centroids) within BAND,
    EDT of the sign beyond, clipped to +-CLIP."""
    v, fc, _, _ = marching_cubes(np.pad(f, 1, constant_values=f.max()), 0.0)
    v -= 1
    pts = np.concatenate([v, v[fc].mean(1)])
    inside = f < 0
    e = np.where(inside, ndimage.distance_transform_edt(inside),
                 ndimage.distance_transform_edt(~inside)) * 0.5
    band = e <= BAND
    dist, _ = cKDTree(pts).query(np.argwhere(band), workers=8)
    sdf = np.where(inside, -(e + 0.0), e).astype(np.float32)
    sdf[band] = np.where(inside[band], -dist, dist) * 0.5
    return np.clip(sdf, -CLIP, CLIP)


def build_targets(mask, labels, aff, dev):
    """{'g05': [lh, rh], 'ivb25': [lh, rh]} SDFs on the 0.5 mm grid of the mask
    bbox + PAD, and that grid's vox2world."""
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - PAD, 0)
    hi = np.minimum(idx.max(0) + PAD + 1, mask.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    aff05 = aff.copy()
    aff05[:3, :3] = aff[:3, :3] * 0.5
    aff05[:3, 3] = (aff @ np.r_[lo, 1.0])[:3]
    hemis = [h[sl] for h in hemi_masks(mask, labels)]
    return {'g05': [redistance(gauss_field(h)) for h in hemis],
            'ivb25': [redistance(ivb_field(h, dev)) for h in hemis]}, aff05


# =============================================================================
# 2. Demons (mesh SDF -> g05)
# =============================================================================

def rasterize(verts, faces, n_lh, shape, affine):
    """Inside mask of the closed mesh at voxel centres, per hemisphere then
    OR-ed: ray parity along array axis 0 (grid nudged by an irrational
    sub-voxel offset so no ray hits a vertex or edge)."""
    out = np.zeros(shape, bool)
    vox = world_to_vox(verts, affine)
    lhf = faces[:, 0] < n_lh
    for v, f in ((vox[:n_lh], faces[lhf]), (vox[n_lh:], faces[~lhf] - n_lh)):
        t = v[f].copy()
        t[:, :, 1] -= 1.37e-4
        t[:, :, 2] -= 2.71e-4
        j0, j1 = np.ceil(t[:, :, 1].min(1)).astype(int), np.floor(t[:, :, 1].max(1)).astype(int)
        k0, k1 = np.ceil(t[:, :, 2].min(1)).astype(int), np.floor(t[:, :, 2].max(1)).astype(int)
        nj, nk = np.maximum(j1 - j0 + 1, 0), np.maximum(k1 - k0 + 1, 0)
        cnt = nj * nk
        fi = np.repeat(np.arange(len(f)), cnt)
        r = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        jj, kk = j0[fi] + r // nk[fi], k0[fi] + r % nk[fi]
        a, b, c = t[fi, 0], t[fi, 1], t[fi, 2]
        den = (b[:, 1] - a[:, 1]) * (c[:, 2] - a[:, 2]) - (c[:, 1] - a[:, 1]) * (b[:, 2] - a[:, 2])
        with np.errstate(divide='ignore', invalid='ignore'):
            w1 = ((jj - a[:, 1]) * (c[:, 2] - a[:, 2]) - (c[:, 1] - a[:, 1]) * (kk - a[:, 2])) / den
            w2 = ((b[:, 1] - a[:, 1]) * (kk - a[:, 2]) - (jj - a[:, 1]) * (b[:, 2] - a[:, 2])) / den
        ok = (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1) & np.isfinite(w1) & \
             (jj >= 0) & (jj < shape[1]) & (kk >= 0) & (kk < shape[2])
        x = a[ok, 0] + w1[ok] * (b[ok, 0] - a[ok, 0]) + w2[ok] * (c[ok, 0] - a[ok, 0])
        i = np.clip(np.ceil(x).astype(int), 0, shape[0])
        flat = (i * shape[1] + jj[ok]) * shape[2] + kk[ok]
        hits = np.bincount(flat, minlength=(shape[0] + 1) * shape[1] * shape[2])
        out |= (np.cumsum(hits.reshape(shape[0] + 1, *shape[1:]), axis=0)[:-1] & 1).astype(bool)
    return out


def point_tri_dist(p, a, b, c):
    """Exact point -> triangle distance, broadcast over leading dims."""
    ab, ac = b - a, c - a
    n = torch.cross(ab, ac, dim=-1)
    nn = (n * n).sum(-1)
    t = ((p - a) * n).sum(-1) / nn.clamp_min(1e-20)
    q = p - t[..., None] * n
    aq = q - a
    d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
    d20, d21 = (aq * ab).sum(-1), (aq * ac).sum(-1)
    den = (d00 * d11 - d01 * d01).clamp_min(1e-20)
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    inside = (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1) & (nn > 1e-14)
    dplane = (p - q).norm(dim=-1)

    def seg(x, y):
        xy = y - x
        s = (((p - x) * xy).sum(-1) / (xy * xy).sum(-1).clamp_min(1e-20)).clamp(0, 1)
        return (p - (x + s[..., None] * xy)).norm(dim=-1)
    de = torch.minimum(torch.minimum(seg(a, b), seg(b, c)), seg(c, a))
    return torch.where(inside, torch.minimum(dplane, de), de)


def mesh_sdf(sv, sf, inside, A05, band, dev, k=12):
    """Mesh SDF (mm, negative inside) on the 0.5 mm grid: sign from `inside`,
    exact point -> triangle distance (over the k nearest face centroids) within
    `band` of the voxelised boundary, +-band beyond."""
    approx = np.where(inside, ndimage.distance_transform_edt(inside) * 0.5,
                      ndimage.distance_transform_edt(~inside) * 0.5)
    sdf = np.where(inside, -band, band).astype(np.float32)
    idx = np.nonzero(approx <= band + 0.75)
    pts = vox_to_world(np.stack(idx, 1), A05)
    _, cand = cKDTree(sv[sf].mean(1)).query(pts, k=k, workers=16)
    tri = torch.as_tensor(sv[sf], dtype=torch.float32, device=dev)
    d = np.empty(len(pts), np.float32)
    for s in range(0, len(pts), 1_000_000):
        P = torch.as_tensor(pts[s:s + 1_000_000], dtype=torch.float32, device=dev)[:, None]
        T = tri[torch.as_tensor(cand[s:s + 1_000_000], device=dev)]
        d[s:s + len(P)] = point_tri_dist(P, T[..., 0, :], T[..., 1, :], T[..., 2, :]).min(1)[0].cpu().numpy()
    sdf[idx] = np.clip(np.where(inside[idx], -1.0, 1.0) * d, -band, band)
    return sdf


def to_grid(disp):
    """Voxel displacement (1, 3, D, H, W) -> grid_sample normalised offsets."""
    D, H, W = disp.shape[2:]
    sc = torch.tensor([2 / (D - 1), 2 / (H - 1), 2 / (W - 1)], device=disp.device)
    return (disp * sc[None, :, None, None, None]).permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]


def ident(shape, dev):
    z, y, x = torch.meshgrid(*[torch.linspace(-1, 1, s, device=dev) for s in shape], indexing='ij')
    return torch.stack([x, y, z], -1)[None]


def warp(img, disp, idg):
    return F.grid_sample(img, idg + to_grid(disp), mode='bilinear',
                         padding_mode='border', align_corners=True)


def expmap(v, idg, n=7):
    """Scaling and squaring: displacement of exp(v)."""
    d = v / 2 ** n
    for _ in range(n):
        d = d + warp(d, d, idg)
    return d


def gauss(x, sigma_vox):
    """Separable Gaussian, replicate padding, per channel."""
    r = int(np.ceil(3 * sigma_vox))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma_vox) ** 2)
    k = k / k.sum()
    C = x.shape[1]
    for ax in range(3):
        shp = [1, 1, 1, 1, 1]
        shp[2 + ax] = len(k)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode='replicate'), k.view(shp).repeat(C, 1, 1, 1, 1), groups=C)
    return x


def grad(img):
    return torch.cat([(torch.roll(img, -1, ax) - torch.roll(img, 1, ax)) / 2 for ax in (2, 3, 4)], 1)


def demons(fixed, moving, dev, iters=(40, 40), sig_fluid=0.5, sig_diff=0.25, kmax=0.5,
           band=3.0):
    """Symmetric-gradient log-domain demons, 1 mm then 0.5 mm: find psi = exp(v)
    with moving(psi(x)) ~ fixed(x). Force where |fixed| or |moving(psi)| is
    within `band` mm; fluid (update) and diffusion (field) Gaussians in mm; step
    capped at kmax voxels. Returns psi's displacement in 0.5 mm voxels."""
    Ft = torch.as_tensor(fixed, device=dev)[None, None]
    Tt = torch.as_tensor(moving, device=dev)[None, None]
    v = None
    for lev, n_it in zip((1, 0), iters):
        vox = 0.5 * 2 ** lev
        f = Ft[..., ::2, ::2, ::2] if lev else Ft
        t = Tt[..., ::2, ::2, ::2] if lev else Tt
        f, t = f / vox, t / vox                         # voxel units -> |grad| ~ 1
        idg = ident(f.shape[2:], dev)
        if v is None:
            v = torch.zeros((1, 3) + tuple(f.shape[2:]), device=dev)
        else:
            v = F.interpolate(v, size=f.shape[2:], mode='trilinear', align_corners=True) * 2
        gf = grad(f)
        fb = band / vox
        near = f.abs() < fb
        for _ in range(n_it):
            w = warp(t, expmap(v, idg), idg)
            diff = w - f
            J = 0.5 * (grad(w) + gf)
            den = (J * J).sum(1, keepdim=True) + diff * diff / kmax ** 2
            u = -diff * J / den.clamp_min(1e-6)
            u = u * (near | (w.abs() < fb)).float()
            v = gauss(v + gauss(u, sig_fluid / vox), sig_diff / vox)
    return expmap(v, ident(fixed.shape, dev))


def demons_step(v, f, mask, aff, g05, aff05, dev, band=3.0, margin=6.0):
    """Both hemispheres in one 0.5 mm box (mesh + mask + margin mm): demons
    from the mesh SDF to the g05 SDF (min over hemispheres), vertices moved
    through psi."""
    p = np.concatenate([world_to_vox(v, aff), np.argwhere(mask).astype(float)])
    lo = np.maximum(np.floor(p.min(0) - margin).astype(int), 0)
    hi = np.minimum(np.ceil(p.max(0) + margin).astype(int), np.array(mask.shape) - 1)
    shp = tuple(2 * (hi - lo) + 1)
    m = np.diag([0.5, 0.5, 0.5, 1.0])
    m[:3, 3] = lo
    A = aff @ m                                     # vox2world of the 0.5 mm box
    fixed = mesh_sdf(v, f, rasterize(v, f, N_LH, shp, A), A, band, dev)
    g = np.stack(np.meshgrid(*[np.arange(s) for s in shp], indexing='ij')).reshape(3, -1).T
    vx = world_to_vox(vox_to_world(g, A), aff05)
    moving = ndimage.map_coordinates(np.minimum(*g05), vx.T, order=1, mode='nearest')
    moving = np.clip(moving.reshape(shp).astype(np.float32), -band, band)
    d = demons(fixed, moving, dev, band=band)
    gv = torch.as_tensor(world_to_vox(v, A), dtype=torch.float32, device=dev)
    D, H, W = shp
    gn = (gv / torch.tensor([D - 1, H - 1, W - 1], device=dev) * 2 - 1)[:, [2, 1, 0]]
    dv = F.grid_sample(d, gn.view(1, -1, 1, 1, 3), mode='bilinear', align_corners=True)
    dv = dv.view(3, -1).T.cpu().numpy().astype(np.float64)
    return vox_to_world(world_to_vox(v, A) + dv, A)


# =============================================================================
# 3. Backtrack + repair
# =============================================================================

def crossing(v, f):
    return R.detect(v, f, quiet=True)[1].any()


def backtrack(v0, v1, f, k=2, max_iter=25, rad=3.0):
    """Scale the move v0 -> v1 back on crossing faces + k rings (halving per
    round, falling off over the rings), re-checking only faces within rad mm."""
    src, dst, _ = R.adjacency(f, len(v0))
    cen_tree = cKDTree(v1[f].mean(1))
    hit = R.detect(v1, f, quiet=True)[1]
    alpha = np.ones(len(v0))
    v = v1.copy()
    for _ in range(max_iter):
        if not hit.any():
            break
        ring = R.rings(np.isin(np.arange(len(v0)), f[hit]), src, dst, k)
        alpha *= np.where(ring >= 0, 0.5 + 0.5 * ring / (k + 1.0), 1.0)
        v = v0 + alpha[:, None] * (v1 - v0)
        ch = np.nonzero((ring[f] >= 0).any(1))[0]
        sub = np.unique(np.concatenate(cen_tree.query_ball_point(v1[f[ch]].mean(1), rad)))
        hs = R.detect(v, f[sub], quiet=True)[1]
        hit = np.zeros(len(f), bool)
        hit[sub[hs]] = True
    return v


# =============================================================================
# 4. Snap and fairing (torch, float64, collision-guarded)
# =============================================================================

def _t(x, dt=torch.float64):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dt)


class Field:
    """Trilinear, edge-clamped sampler of one volume at world points."""

    def __init__(self, vol, aff):
        self.v = _t(vol, torch.float32)
        inv = np.linalg.inv(aff)
        self.A, self.b = _t(inv[:3, :3]), _t(inv[:3, 3])
        self.hi = torch.tensor(self.v.shape, dtype=torch.float64) - 1

    def __call__(self, p):
        x = torch.minimum((p @ self.A.T + self.b).clamp(min=0), self.hi)
        i0 = torch.minimum(x.floor(), self.hi - 1).long()
        w = (x - i0).float()
        out = 0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ww = ((w[:, 0] if dx else 1 - w[:, 0]) * (w[:, 1] if dy else 1 - w[:, 1])
                          * (w[:, 2] if dz else 1 - w[:, 2]))
                    out = out + ww * self.v[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        return out.double()


class HemiSDF:
    """lh vertices [0, n_lh) read the lh volume, the rest the rh volume."""

    def __init__(self, vols, aff, n_lh):
        self.f, self.n_lh = [Field(v, aff) for v in vols], n_lh

    def __call__(self, p):
        return torch.cat([self.f[0](p[:self.n_lh]), self.f[1](p[self.n_lh:])])

    def grad(self, p, h=0.25):
        g = torch.zeros_like(p)
        for c in range(3):
            e = torch.zeros(3, dtype=p.dtype)
            e[c] = h
            g[:, c] = (self(p + e) - self(p - e)) / (2 * h)
        return g


def normals(v, f):
    t = v[f]
    fn = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=1)
    vn = torch.zeros_like(v).index_add_(0, f.reshape(-1), fn.repeat_interleave(3, 0))
    return vn / vn.norm(dim=1, keepdim=True).clamp(min=1e-30)


def adjacency(f, n):
    e = f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    e = torch.cat([e, e.flip(1)])
    key = torch.unique(e[:, 0] * n + e[:, 1])
    src, dst = key // n, key % n
    return src, dst, torch.bincount(src, minlength=n).clamp(min=1).to(torch.float64)


def umbrella(v, src, dst, deg):
    return torch.zeros_like(v).index_add_(0, src, v[dst]) / deg[:, None] - v


def _seg_hits(P, Q, T):
    D, e1, e2 = Q - P, T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    pv = torch.cross(D, e2, dim=1)
    det = (e1 * pv).sum(1)
    ok = det.abs() >= 1e-14
    inv = torch.where(ok, 1.0 / torch.where(ok, det, torch.ones_like(det)), torch.zeros_like(det))
    tv = P - T[:, 0]
    qv = torch.cross(tv, e1, dim=1)
    u, w, t = (tv * pv).sum(1) * inv, (D * qv).sum(1) * inv, (e2 * qv).sum(1) * inv
    return ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t >= 0) & (t <= 1)


def _tri_cross(A, B):
    hit = torch.zeros(len(A), dtype=torch.bool)
    for X, Y in ((A, B), (B, A)):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            hit |= _seg_hits(X[:, a], X[:, b], Y)
    return hit


def detect(v, f, cell=1.0, active=None, chunk=4_000_000):
    """Faces crossing a non-adjacent face (torch port of
    repair_template_mesh.detect). `active`: only pairs with an active face are
    tested — a move from a clean state can only create crossings that involve
    a moved face."""
    tri = v[f]
    lo, hi = tri.min(1).values, tri.max(1).values
    c_lo, c_hi = torch.floor(lo / cell).long(), torch.floor(hi / cell).long()
    span = c_hi - c_lo + 1
    cnt = span.prod(1)
    tri_id = torch.repeat_interleave(torch.arange(len(f)), cnt)
    k = torch.arange(int(cnt.sum())) - torch.repeat_interleave(cnt.cumsum(0) - cnt, cnt)
    nx, ny = span[tri_id, 0], span[tri_id, 1]
    g = c_lo[tri_id] + torch.stack([k % nx, (k // nx) % ny, k // (nx * ny)], 1) - c_lo.min(0).values
    dim = c_hi.max(0).values - c_lo.min(0).values + 1
    key = (g[:, 2] * dim[1] + g[:, 1]) * dim[0] + g[:, 0]
    if active is not None:                      # drop cells holding no active face
        keep = torch.isin(key, torch.unique(key[active[tri_id]]))
        key, tri_id = key[keep], tri_id[keep]
    key, order = torch.sort(key, stable=True)
    tri_s = tri_id[order]
    _, size = torch.unique_consecutive(key, return_counts=True)
    start = size.cumsum(0) - size
    hit = torch.zeros(len(f), dtype=torch.bool)
    for s in torch.unique(size[size > 1]).tolist():
        st = start[size == s]
        a, b = torch.triu_indices(s, s, 1)
        step = max(1, chunk // len(a))
        for c0 in range(0, len(st), step):
            blk = tri_s[st[c0:c0 + step][:, None] + torch.arange(s)]
            i, j = blk[:, a].reshape(-1), blk[:, b].reshape(-1)
            m = (lo[i] <= hi[j]).all(1) & (lo[j] <= hi[i]).all(1)
            if active is not None:
                m &= active[i] | active[j]
            i, j = i[m], j[m]
            m = ~(f[i][:, :, None] == f[j][:, None, :]).any(2).any(1)
            i, j = i[m], j[m]
            x = _tri_cross(tri[i], tri[j])
            hit[i[x]] = True
            hit[j[x]] = True
    return hit


def snap(verts, faces, sdf, move_mask, n_iter, step, lam_t, blend, cap, min_gain,
         check_every):
    """Active surface onto the zero level of `sdf`: each step every vertex
    moves by -clip(sdf, +-step) along a blend of its normal and the SDF
    gradient (move_mask False -> no SDF force) plus lam_t of the tangential
    umbrella; total move capped at `cap` mm. Every check_every steps the
    vertices of crossing faces go back to the last clean state and their step
    gain halves (< min_gain -> frozen)."""
    v0 = _t(verts)
    f = _t(faces, torch.long)
    n = len(v0)
    src, dst, deg = adjacency(f, n)
    mm = _t(move_mask, torch.bool).double()
    gain = torch.ones(n, dtype=torch.float64)
    v, ck = v0.clone(), v0.clone()             # ck: last crossing-free positions
    for it in range(n_iter):
        nrm = normals(v, f)
        d = sdf(v)
        mag = d.abs().clamp(max=step) * torch.where(d > 0, 1.0, -1.0) * mm
        g = sdf.grad(v)
        g = g / g.norm(dim=1, keepdim=True).clamp(min=1e-6)
        dirn = (1 - blend) * nrm + blend * g
        dirn = dirn / dirn.norm(dim=1, keepdim=True).clamp(min=1e-6)
        L = umbrella(v, src, dst, deg)
        Ln = (L * nrm).sum(1, keepdim=True) * nrm
        v = v + (-mag[:, None] * dirn + lam_t * (L - Ln)) * gain[:, None]
        disp = v - v0
        dn = disp.norm(dim=1, keepdim=True)
        v = torch.where(dn > cap, v0 + disp * (cap / dn.clamp(min=1e-12)), v)
        if (it + 1) % check_every == 0 or it == n_iter - 1:
            moved = (v - ck).norm(dim=1) > 0
            for _ in range(8):
                hit = detect(v, f, active=moved[f].any(1))
                if not hit.any():
                    break
                bad = torch.zeros(n, dtype=torch.bool)
                bad[f[hit].reshape(-1)] = True
                v[bad] = ck[bad]
                gain[bad] *= 0.5
                moved = bad
            else:                                # still crossing: drop the whole chunk
                v = ck.clone()
            gain[gain < min_gain] = 0
            ck = v.clone()
    return v.numpy()


def fair(verts, faces, n_iter=5, lam=0.2, check_every=2, ring=1):
    """Tangential umbrella smoothing (triangle quality, no shrinkage); every
    check_every steps the vertices of crossing faces (+ ring rings) go back to
    the last clean state and are frozen."""
    vv, ft = _t(verts), torch.as_tensor(faces)
    n = len(vv)
    src, dst, deg = adjacency(ft, n)
    gain = torch.ones(n, dtype=torch.float64)
    ck = vv.clone()
    for it in range(n_iter):
        L = umbrella(vv, src, dst, deg)
        nrm = normals(vv, ft)
        L = L - (L * nrm).sum(1, keepdim=True) * nrm
        vv = vv + lam * L * gain[:, None]
        if (it + 1) % check_every == 0 or it == n_iter - 1:
            moved = (vv - ck).norm(dim=1) > 0
            for _ in range(10):
                hit = detect(vv, ft, active=moved[ft].any(1))
                if not hit.any():
                    break
                bad = torch.zeros(n, dtype=torch.bool)
                bad[ft[hit].reshape(-1)] = True
                for _ in range(ring):
                    bad[dst[bad[src]]] = True
                vv[bad] = ck[bad]
                gain[bad] = 0
                moved = bad
            else:
                vv = ck.clone()
            ck = vv.clone()
    return vv.numpy()


# =============================================================================
# Chain
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--surf', required=True, help='start mesh, lh + rh in one .surf')
    ap.add_argument('--input_seg', required=True,
                    help='SynthSeg one-hot .npy (class 3 = white-surface interior)')
    ap.add_argument('--synthseg', required=True,
                    help='SynthSeg label volume on the same grid (hemisphere split, affine)')
    ap.add_argument('--cortex_labels_dir', required=True,
                    help='fsaverage-164k nomedialwall label GIFTIs (no snap force on '
                         'the medial wall)')
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda:0', help='targets + demons; snap runs on CPU')
    ap.add_argument('--threads', type=int, default=3, help='torch CPU threads for snap')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    t0 = time.time()
    tick = lambda s: print(f'[cleanup] {time.time() - t0:6.1f} s  {s}', flush=True)

    img = nib.load(args.synthseg)
    aff, labels = img.affine, np.asanyarray(img.dataobj).astype(np.int16)
    mask = np.load(args.input_seg, mmap_mode='r')[WM].astype(np.uint8) > 0
    cras = apply_affine(aff, np.array(img.shape[:3], float) / 2.0)
    coords, faces = nib.freesurfer.read_geometry(args.surf)
    if len(coords) != 2 * N_LH:
        raise SystemExit(f'[cleanup] {args.surf}: {len(coords)} vertices, expected 2 x {N_LH}')
    v_start = coords.astype(np.float64) + cras
    f = faces.astype(np.int64)
    cortex = load_cortex_mask(args.cortex_labels_dir)

    targets, aff05 = build_targets(mask, labels, aff, dev)
    tick('targets g05 + ivb25')
    v_dem = as_saved(demons_step(v_start, f, mask, aff, targets['g05'], aff05, dev), cras)
    tick(f'demons (max move {np.linalg.norm(v_dem - v_start, axis=1).max():.2f} mm)')
    v = backtrack(v_start, v_dem, f) if crossing(v_dem, f) else v_dem
    v = as_saved(R.repair(v, f)[0] if crossing(v, f) else v, cras)
    tick('backtrack + repair')
    ivb = HemiSDF(targets['ivb25'], aff05, N_LH)
    v = as_saved(snap(v, f, ivb, cortex, n_iter=40, **SNAP), cras)
    tick('snap 40')
    v = as_saved(fair(v, f), cras)
    tick('tangential fairing x5')
    v = as_saved(snap(v, f, ivb, cortex, n_iter=20, **SNAP), cras)
    tick('snap 20')
    if detect(_t(v), _t(f, torch.long)).any():
        v = R.repair(v, f)[0]
    tick(f"final repair | self-intersecting faces {R.detect(v, f, quiet=True)[0]['si_faces']}")
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_git_sha(out.parent)
    nib.freesurfer.write_geometry(str(out), (v - cras).astype(np.float32), f.astype(np.int32))
    tick(f'wrote {out}')


if __name__ == '__main__':
    main()
