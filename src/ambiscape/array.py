"""Spaced-microphone array analysis: TDOAs, bearings, coherence, triangulation.

The toolbox's three spatial paradigms differ in where the microphones sit.
:mod:`ambiscape.spatial` reads a soundfield sampled at one point (co-located
ambisonic capsules) and asks *from which direction*; :mod:`ambiscape.network`
reads one microphone per room and asks *through which fabric*; ``array`` sits
between them — a handful of spaced omnis in one room (the SINS nodes' linear
four-MEMS arrays) whose wavefront arrival-time differences and inter-channel
coherence carry direction and diffuseness where no soundfield microphone was
present.

- ``load_geometry`` — mic positions in metres from inline coordinates or a
  small JSON file;
- ``tdoa`` — pairwise GCC-PHAT time-difference-of-arrival per frame, with
  the GCC peak height and its prominence over the runner-up;
- ``bearing`` — frame-wise bearing for a *linear* array from a weighted
  least-squares fit across the pair TDOAs, with a confidence stream;
- ``coherence_profile`` — inter-channel magnitude-squared coherence versus
  frequency per window, against the analytic diffuse-field curve for each
  spacing, and the per-window diffuseness proxy ``gamma_array``;
- ``triangulate`` — least-squares intersection of bearing streams from two
  or more nodes on a floor plan, to coarse source positions;
- ``bearing_figure`` / ``coherence_figure`` / ``triangulate_figure`` — the
  matching figures;
- ``run_array`` — CLI driver: one multichannel WAV in, JSON + figures out.

Conventions. ``tau_s[frame, pair] = t_i - t_j`` for pair ``(i, j)``:
positive when the wavefront reaches mic ``j`` first. Bearings are measured
from the array axis (the unit vector from the first mic towards the last):
0° and 180° are the endfire directions, 90° is broadside. A linear array
cannot tell the two sides of its axis apart (front–back ambiguity: only the
cone angle is observable), and near endfire the bearing loses resolution
because the delay–angle mapping flattens (``d tau / d theta -> 0``), so
endfire estimates are wide even at high confidence. ``gamma_array`` is a
direct/diffuse *proxy* built from coherence deviations between spaced
omnis; it is deliberately named apart from the first-order-ambisonic
diffuseness ``psi`` (``diffuse`` in the feature cache), which is an
energetic soundfield measure at a single point — the two agree in tendency,
not in value.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy import signal

EPS = 1e-12
C_SOUND = 343.0                       # default speed of sound (m/s, ~20 °C)


# ---------------------------------------------------------------- geometry


def load_geometry(geometry) -> dict:
    """Microphone geometry: positions in metres + speed of sound.

    Accepts a dict, a path to a JSON file, or a bare coordinate sequence.
    The JSON/dict form is ``{"mics": [[x, y], ...], "c": 343.0}`` (``c``
    optional); ``mics`` may also be a flat list of numbers, read as
    positions along one line. Returns ``{"pos" (n, 2), "c", "linear",
    "axis"}`` where ``axis`` is the unit vector from the first mic towards
    the last (the bearing reference) and ``linear`` says whether all mics
    sit on one line — the geometry :func:`bearing` requires.
    """
    if isinstance(geometry, (str, Path)):
        geometry = json.loads(Path(geometry).read_text())
    if isinstance(geometry, dict) and "pos" in geometry:
        return geometry                        # already loaded
    if isinstance(geometry, dict):
        mics, c = geometry.get("mics"), float(geometry.get("c", C_SOUND))
    else:
        mics, c = geometry, C_SOUND
    pos = np.asarray(mics, float)
    if pos.ndim == 1:                          # distances along one line
        pos = np.stack([pos, np.zeros_like(pos)], 1)
    if pos.ndim != 2 or pos.shape[1] != 2 or pos.shape[0] < 2:
        raise ValueError("geometry needs at least two [x, y] mic positions "
                         "in metres (or a flat list of on-axis positions)")
    span = pos - pos.mean(0)
    sv = np.linalg.svd(span, compute_uv=False)
    linear = bool(sv[1] <= 1e-9 + 1e-4 * sv[0])
    axis = pos[-1] - pos[0]
    axis = axis / (np.linalg.norm(axis) + EPS)
    return {"pos": pos, "c": c, "linear": linear, "axis": axis}


def _pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


# ---------------------------------------------------------------- GCC-PHAT


def tdoa(data: np.ndarray, fs: int, geometry, frame_s: float = 0.1,
         hop_s: float | None = None) -> dict:
    """Pairwise GCC-PHAT time differences of arrival per frame.

    ``data`` is ``(samples, channels)`` with one channel per geometry mic.
    Each frame (``frame_s`` long, hopped by ``hop_s``, default half a
    frame) is Hann-windowed, and for every mic pair the PHAT-weighted
    cross-power spectrum is inverted to a generalised cross-correlation;
    the peak is searched only over physically possible lags (± spacing /
    ``c``, a small margin added) and refined by parabolic interpolation.

    Returns ``{"t", "pairs", "d_m", "tau_s", "peak", "prominence"}``:
    frame centres in seconds, the ``(i, j)`` pair list with spacings,
    ``tau_s[frame, pair] = t_i - t_j`` (positive = the wavefront reaches
    mic ``j`` first), the GCC peak height (1 for a perfect single delay,
    near 0 for decorrelated channels), and the peak's prominence over the
    strongest rival lag outside ±3 samples — the confidence base.
    """
    g = load_geometry(geometry)
    x = np.asarray(data, np.float64)
    if x.ndim != 2:
        raise ValueError("data must be (samples, channels)")
    n, nch = x.shape
    if nch != len(g["pos"]):
        raise ValueError(f"recording has {nch} channels but the geometry "
                         f"has {len(g['pos'])} mics")
    L = int(round(frame_s * fs))
    hop = int(round((hop_s if hop_s else frame_s / 2) * fs))
    if n < L:
        raise ValueError("recording shorter than one analysis frame")
    nf = 1 + (n - L) // hop
    win = np.hanning(L)
    nfft = 1 << int(np.ceil(np.log2(2 * L)))
    pairs = _pairs(nch)
    d = np.array([float(np.linalg.norm(g["pos"][i] - g["pos"][j]))
                  for i, j in pairs])
    K = np.clip(np.ceil(d / g["c"] * fs).astype(int) + 2, 2, nfft // 2 - 1)
    tau = np.full((nf, len(pairs)), np.nan)
    peak = np.zeros((nf, len(pairs)))
    prom = np.zeros((nf, len(pairs)))
    for f0 in range(0, nf, 256):                 # bounded memory per block
        f1 = min(f0 + 256, nf)
        idx = np.arange(L)[None, :] + hop * np.arange(f0, f1)[:, None]
        X = [np.fft.rfft(x[:, ch][idx] * win, nfft) for ch in range(nch)]
        rows = np.arange(f1 - f0)
        for p, (i, j) in enumerate(pairs):
            R = X[i] * np.conj(X[j])
            R /= np.abs(R) + EPS
            cc = np.fft.irfft(R, nfft)
            k = K[p]
            w = np.concatenate([cc[:, -k:], cc[:, :k + 1]], 1)  # lags -k..k
            pk = np.argmax(w, 1)
            v = w[rows, pk]
            lo = w[rows, np.maximum(pk - 1, 0)]
            hi = w[rows, np.minimum(pk + 1, 2 * k)]
            den = lo - 2 * v + hi
            adj = np.where((pk > 0) & (pk < 2 * k) & (den < 0),
                           0.5 * (lo - hi) / np.where(den == 0, 1.0, den),
                           0.0)
            tau[f0:f1, p] = (pk + adj - k) / fs
            peak[f0:f1, p] = v
            for off in range(-3, 4):             # blank the peak, keep rivals
                w[rows, np.clip(pk + off, 0, 2 * k)] = -np.inf
            prom[f0:f1, p] = np.clip(v - np.maximum(w.max(1), 0.0), 0.0, 1.0)
    return {"t": (L / 2 + hop * np.arange(nf)) / fs, "pairs": pairs,
            "d_m": d, "tau_s": tau, "peak": peak, "prominence": prom,
            "frame_s": L / fs, "hop_s": hop / fs, "fs": int(fs),
            "geometry": g}


# ---------------------------------------------------------------- bearing


def bearing(td: dict, geometry=None) -> dict:
    """Frame-wise bearing of a linear array from the pair TDOAs.

    For mics on one line with axis ``u``, a plane wave from cone angle
    ``theta`` (measured from ``u``) gives ``tau_ij = -(s_i - s_j) *
    cos(theta) / c`` with ``s`` the on-axis positions, so each frame's
    ``cos(theta)`` is a prominence-weighted least-squares fit across all
    pairs. Returns ``{"t", "bearing_deg", "cos_theta", "confidence",
    "residual_s", "clipped"}``: bearings in [0, 180]° (0/180 = endfire,
    90 = broadside — the two sides of the axis are indistinguishable),
    the median GCC prominence across pairs as confidence, the RMS TDOA
    residual of the fit, and a flag for frames whose fitted ``cos(theta)``
    fell outside [-1, 1] (clipped to the nearest endfire; treat those
    bearings as unreliable whatever the confidence says).
    """
    g = td.get("geometry") or load_geometry(geometry)
    if not g["linear"]:
        raise ValueError("bearing requires a linear array — for other "
                         "layouts work from the pair TDOAs directly")
    s = g["pos"] @ g["axis"]
    ds = np.array([s[i] - s[j] for i, j in td["pairs"]])   # signed spacing
    w = np.maximum(td["prominence"], 0.0)
    tau = np.where(np.isfinite(td["tau_s"]), td["tau_s"], 0.0)
    denom = (w * ds ** 2).sum(1)
    num = -g["c"] * (w * ds * tau).sum(1)
    valid = denom > EPS
    cos_t = np.where(valid, num / np.where(valid, denom, 1.0), np.nan)
    clipped = np.abs(cos_t) > 1.0
    cos_c = np.clip(cos_t, -1.0, 1.0)
    resid = np.sqrt((w * (tau + ds * cos_c[:, None] / g["c"]) ** 2).sum(1)
                    / (w.sum(1) + EPS))
    return {"t": td["t"], "bearing_deg": np.degrees(np.arccos(cos_c)),
            "cos_theta": cos_t,
            "confidence": np.median(td["prominence"], 1),
            "residual_s": resid, "clipped": clipped}


def bearing_figure(b: dict, out_path: str | Path,
                   title: str | None = None) -> Path:
    """Bearing track: time × bearing, colour = confidence.

    Endfire rows (0° and 180°) and broadside (90°) are marked; the scatter
    fades with the GCC-prominence confidence, so unreliable frames recede.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=130)
    conf = np.clip(b["confidence"], 0, 1)
    sc = ax.scatter(b["t"], b["bearing_deg"], c=conf, cmap="viridis",
                    s=6, vmin=0, vmax=max(0.5, float(conf.max()) or 0.5),
                    alpha=0.85, linewidths=0)
    for y, lab in ((0, "endfire"), (90, "broadside"), (180, "endfire")):
        ax.axhline(y, color="0.8", lw=0.7, ls="--")
        ax.text(1.002, y / 180, lab, transform=ax.transAxes, fontsize=7,
                va="center", color="0.4")
    ax.set(xlabel="time (s)", ylabel="bearing from array axis (°)",
           ylim=(-5, 185), yticks=(0, 45, 90, 135, 180),
           title=(title or "") + " — bearing track (front–back ambiguous; "
           "sides of the axis fold together)")
    fig.colorbar(sc, ax=ax, pad=0.06, label="confidence (GCC prominence)")
    ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------- coherence


def coherence_profile(data: np.ndarray, fs: int, geometry,
                      win_s: float = 4.0, nperseg: int | None = None) -> dict:
    """Inter-channel coherence versus frequency per window, and diffuseness.

    In each non-overlapping ``win_s`` window the magnitude-squared
    coherence of every mic pair is estimated (Welch, ``nperseg`` samples,
    default a power of two near one eighth of the window so each estimate
    averages ~15 segments; fewer segments bias coherence upward). The
    analytic diffuse-field curve for spaced omnis, ``sinc(2 f d / c)^2``,
    is attached per pair.

    Diffuseness: per window and pair, the energy-weighted mean over the
    informative band (where the diffuse curve has fallen below 0.5;
    weights are the pair's cross-channel spectrum level, so bands the
    scene does not excite carry no vote) of the measured coherence's
    excess over the diffuse curve, normalised to [0, 1], gives a
    directness reading; ``gamma_array`` is one minus its median across
    pairs. 0 = a coherent wavefront crosses the array (direct field),
    1 = coherence at or below the diffuse prediction. This is a PROXY
    built from spaced-omni coherence — kept deliberately distinct from
    the ambisonic diffuseness ``psi``, which measures the energetic
    isotropy of a soundfield at one point. Pairs too closely spaced to
    fall below 0.5 within the bandwidth contribute NaN.

    Returns ``{"win_t", "f", "msc" (nwin, npairs, nf), "msc_diffuse"
    (npairs, nf), "gamma_pair" (nwin, npairs), "gamma_array" (nwin,),
    "pairs", "d_m"}``.
    """
    g = load_geometry(geometry)
    x = np.asarray(data, np.float64)
    n, nch = x.shape
    if nch != len(g["pos"]):
        raise ValueError(f"recording has {nch} channels but the geometry "
                         f"has {len(g['pos'])} mics")
    W = int(round(win_s * fs))
    if n < W:                       # short take: measure the whole of it
        W = n
    nwin = n // W
    nper = int(nperseg) if nperseg else \
        int(np.clip(1 << int(np.log2(max(W / 8, 2))), 128, 1024))
    pairs = _pairs(nch)
    d = np.array([float(np.linalg.norm(g["pos"][i] - g["pos"][j]))
                  for i, j in pairs])
    f = np.fft.rfftfreq(nper, 1 / fs)
    msc = np.zeros((nwin, len(pairs), len(f)))
    lvl = np.zeros((nwin, len(pairs), len(f)))     # pair spectrum level
    for wi in range(nwin):
        seg = x[wi * W:(wi + 1) * W]
        psd = np.stack([signal.welch(seg[:, ch], fs, nperseg=nper)[1]
                        for ch in range(nch)])
        for p, (i, j) in enumerate(pairs):
            _, msc[wi, p] = signal.coherence(seg[:, i], seg[:, j], fs,
                                             nperseg=nper)
            lvl[wi, p] = np.sqrt(psd[i] * psd[j])
    msc_diff = np.sinc(2 * f[None, :] * d[:, None] / g["c"]) ** 2
    informative = (msc_diff < 0.5) & (f[None, :] > 0)
    excess = np.clip((msc - msc_diff[None]) / (1 - msc_diff[None] + EPS),
                     0.0, 1.0)
    gamma_pair = np.full((nwin, len(pairs)), np.nan)
    for p in range(len(pairs)):
        if informative[p].any():
            wgt = lvl[:, p, informative[p]]
            wgt = wgt / (wgt.sum(1, keepdims=True) + EPS)
            gamma_pair[:, p] = 1.0 - (excess[:, p, informative[p]]
                                      * wgt).sum(1)
    gamma = np.nanmedian(gamma_pair, 1) if np.isfinite(gamma_pair).any() \
        else np.full(nwin, np.nan)
    return {"win_t": W / fs * (np.arange(nwin) + 0.5), "f": f, "msc": msc,
            "msc_diffuse": msc_diff, "gamma_pair": gamma_pair,
            "gamma_array": gamma, "pairs": pairs, "d_m": d,
            "win_s": W / fs, "nperseg": nper}


def coherence_figure(c: dict, out_path: str | Path,
                     title: str | None = None) -> Path:
    """Coherence profile (median over windows, per pair, with the analytic
    diffuse curves dashed) over the ``gamma_array`` timeline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.2), dpi=130)
    med = np.median(c["msc"], 0)
    cmap = plt.get_cmap("viridis")
    order = np.argsort(c["d_m"])
    for rank, p in enumerate(order):
        col = cmap(rank / max(len(order) - 1, 1) * 0.85)
        i, j = c["pairs"][p]
        ax[0].plot(c["f"], med[p], color=col, lw=1.2,
                   label=f"{i}–{j} ({100 * c['d_m'][p]:.0f} cm)")
        ax[0].plot(c["f"], c["msc_diffuse"][p], color=col, lw=0.8, ls="--")
    ax[0].set(xlabel="frequency (Hz)", ylabel="MSC", ylim=(0, 1.02),
              title=(title or "") + " — coherence vs the diffuse-field "
              "curve (dashed) per spacing")
    ax[0].legend(frameon=False, fontsize=7, ncol=2, title="pair")
    ax[0].grid(alpha=0.2, lw=0.5)
    ax[1].plot(c["win_t"], c["gamma_array"], ".-", ms=4, lw=1.0,
               color="#2a78d6")
    ax[1].set(xlabel="time (s)", ylabel="gamma_array", ylim=(-0.02, 1.02),
              title="diffuseness proxy (0 = direct wavefront, 1 = diffuse)")
    ax[1].grid(alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------- floor plan


def load_plan(plan) -> list[dict]:
    """Floor plan: node positions and array orientations in one frame.

    Accepts a dict, a JSON file path, or a list of node dicts. Format::

        {"nodes": [
          {"name": "living",  "pos": [0.0, 0.0], "axis_deg": 0.0},
          {"name": "kitchen", "pos": [4.0, 3.0], "axis_deg": 90.0}
        ]}

    ``pos`` in metres in the floor-plan frame; ``axis_deg`` is the world
    direction of the node's array axis (first mic towards last),
    anticlockwise from +x. Returns the node list with ``pos`` as arrays.
    """
    if isinstance(plan, (str, Path)):
        plan = json.loads(Path(plan).read_text())
    nodes = plan["nodes"] if isinstance(plan, dict) else plan
    out = []
    for k, nd in enumerate(nodes):
        out.append({"name": str(nd.get("name", f"node{k}")),
                    "pos": np.asarray(nd["pos"], float),
                    "axis_deg": float(nd["axis_deg"])})
    if len(out) < 2:
        raise ValueError("triangulation needs at least two nodes")
    return out


def triangulate(bearings: list[dict], plan, grid_s: float = 1.0,
                min_conf: float = 0.2) -> dict:
    """Least-squares intersection of node bearing streams on a floor plan.

    ``bearings`` holds one :func:`bearing` result per plan node, in plan
    order, on a shared clock. On a common ``grid_s`` grid, each node
    contributes its nearest confident frame (``confidence >= min_conf``,
    not clipped); each linear-array bearing then admits two world rays,
    ``axis_deg ± bearing`` (the front–back ambiguity), and every sign
    combination is solved as a weighted least-squares line intersection.
    The combination with the smallest RMS perpendicular residual wins;
    rays that would place the source *behind* a node are rejected. When
    the runner-up combination fits almost as well — mirror-symmetric
    layouts, e.g. parallel array axes, make the two sides genuinely
    indistinguishable — the point is flagged ``ambiguous`` and the
    front–back ambiguity is NOT resolved; only geometry that breaks the
    symmetry can resolve it.

    Returns ``{"t", "xy" (n, 2), "residual_m", "angle_deg", "confidence",
    "ambiguous", "nodes"}``: grid times with a solution, positions,
    the RMS ray-to-point distance (the coarse uncertainty; treat the
    position as a blob of about that radius), the intersection angle
    between the two best-separated rays (small = poorly conditioned),
    the minimum node confidence used, and the ambiguity flag.
    """
    nodes = load_plan(plan)
    if len(bearings) != len(nodes):
        raise ValueError("one bearing stream per plan node, in plan order")
    t0 = max(float(b["t"][0]) for b in bearings)
    t1 = min(float(b["t"][-1]) for b in bearings)
    if t1 < t0:
        raise ValueError("bearing streams do not overlap in time")
    grid = t0 + grid_s * np.arange(int((t1 - t0) / grid_s) + 1)
    out = {k: [] for k in ("t", "xy", "residual_m", "angle_deg",
                           "confidence", "ambiguous")}
    for tg in grid:
        theta, conf, ok = [], [], True
        for b in bearings:
            k = int(np.argmin(np.abs(b["t"] - tg)))
            if (abs(b["t"][k] - tg) > grid_s / 2
                    or b["confidence"][k] < min_conf or b["clipped"][k]
                    or not np.isfinite(b["bearing_deg"][k])):
                ok = False
                break
            theta.append(float(b["bearing_deg"][k]))
            conf.append(float(b["confidence"][k]))
        if not ok:
            continue
        best = None
        for signs in product((1.0, -1.0), repeat=len(nodes)):
            us, res_n = [], 0.0
            A = np.zeros((2, 2))
            rhs = np.zeros(2)
            for nd, th, sg, w in zip(nodes, theta, signs, conf):
                phi = np.radians(nd["axis_deg"] + sg * th)
                u = np.array([np.cos(phi), np.sin(phi)])
                P = np.eye(2) - np.outer(u, u)
                A += w * P
                rhs += w * P @ nd["pos"]
                us.append(u)
            if abs(np.linalg.det(A)) < 1e-9:      # all rays parallel
                continue
            xy = np.linalg.solve(A, rhs)
            r2, wsum, behind = 0.0, 0.0, False
            for nd, u, w in zip(nodes, us, conf):
                v = xy - nd["pos"]
                if v @ u < 0:
                    behind = True
                r2 += w * float(v @ v - (v @ u) ** 2)
                wsum += w
            resid = np.sqrt(max(r2, 0.0) / wsum) + (1e6 if behind else 0.0)
            cross = min(abs(np.degrees(np.arcsin(np.clip(
                ua[0] * ub[1] - ua[1] * ub[0], -1, 1))))
                for a, ua in enumerate(us) for ub in us[a + 1:]) \
                if len(us) > 1 else 0.0
            cand = (resid, xy, cross)
            if best is None or resid < best[0][0]:
                best = (cand, best[0] if best else None)
            elif best[1] is None or resid < best[1][0]:
                best = (best[0], cand)
        if best is None or best[0][0] >= 1e6:
            continue
        resid, xy, cross = best[0]
        second = best[1][0] if best[1] else np.inf
        out["t"].append(float(tg))
        out["xy"].append(xy)
        out["residual_m"].append(float(resid))
        out["angle_deg"].append(float(cross))
        out["confidence"].append(min(conf))
        out["ambiguous"].append(bool(second <= 2.0 * resid + 0.05))
    return {"t": np.array(out["t"]),
            "xy": (np.array(out["xy"]).reshape(-1, 2)),
            "residual_m": np.array(out["residual_m"]),
            "angle_deg": np.array(out["angle_deg"]),
            "confidence": np.array(out["confidence"]),
            "ambiguous": np.array(out["ambiguous"], bool),
            "nodes": [nd["name"] for nd in nodes]}


def triangulate_figure(tri: dict, plan, out_path: str | Path,
                       title: str | None = None) -> Path:
    """Floor-plan scatter: node arrays with their axes, triangulated
    positions coloured by time; ambiguous fixes drawn hollow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nodes = load_plan(plan)
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=130)
    for nd in nodes:
        u = np.array([np.cos(np.radians(nd["axis_deg"])),
                      np.sin(np.radians(nd["axis_deg"]))])
        ax.annotate("", xy=nd["pos"] + 0.4 * u, xytext=nd["pos"] - 0.4 * u,
                    arrowprops={"arrowstyle": "-|>", "color": "0.35",
                                "lw": 1.6})
        ax.plot(*nd["pos"], "s", ms=8, color="#1a4f8f")
        ax.annotate(nd["name"], nd["pos"], textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    if len(tri["t"]):
        solid = ~tri["ambiguous"]
        if solid.any():
            sc = ax.scatter(tri["xy"][solid, 0], tri["xy"][solid, 1],
                            c=tri["t"][solid], cmap="viridis", s=18,
                            zorder=3)
            fig.colorbar(sc, ax=ax, shrink=0.8, label="time (s)")
        if (~solid).any():
            ax.scatter(tri["xy"][~solid, 0], tri["xy"][~solid, 1],
                       facecolors="none", edgecolors="0.5", s=18, zorder=3,
                       label="ambiguous (mirror fits equally)")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set(xlabel="x (m)", ylabel="y (m)",
           title=(title or "") + " — triangulated positions "
           "(arrows = array axes)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------- runner


def _r(v, nd: int = 3):
    v = float(v)
    return round(v, nd) if np.isfinite(v) else None


def run_array(recording: str | Path, geometry, out_dir: str | Path | None
              = None, frame_s: float = 0.1, hop_s: float | None = None,
              win_s: float = 4.0, min_conf: float = 0.2) -> dict:
    """Analyse one multichannel array recording; write JSON + figures.

    Runs :func:`tdoa`, :func:`bearing` (linear geometries) and
    :func:`coherence_profile` on ``recording``, and writes ``array.json``,
    ``array_bearing.png`` and ``array_coherence.png`` to ``out_dir``
    (default ``<recording dir>/analysis``). Bearing summary statistics
    pool only the confident, unclipped frames (``confidence >=
    min_conf``). Returns the ``array.json`` document. Triangulation
    across several nodes is a library call — see :func:`triangulate`.
    """
    import soundfile as sf
    recording = Path(recording)
    out = Path(out_dir) if out_dir else recording.parent / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    data, fs = sf.read(str(recording), dtype="float64", always_2d=True)
    g = load_geometry(geometry)
    td = tdoa(data, fs, g, frame_s=frame_s, hop_s=hop_s)
    cp = coherence_profile(data, fs, g, win_s=win_s)
    name = recording.stem
    doc = {"recording": recording.name, "n_mics": len(g["pos"]),
           "mics_m": [[round(float(v), 4) for v in p] for p in g["pos"]],
           "c_m_s": g["c"], "linear": g["linear"],
           "params": {"frame_s": td["frame_s"], "hop_s": td["hop_s"],
                      "coherence_win_s": cp["win_s"],
                      "nperseg": cp["nperseg"], "min_conf": min_conf},
           "n_frames": len(td["t"]),
           "tdoa_median_ms": {f"{i}-{j}": _r(1e3 * v)
                              for (i, j), v in zip(
                                  td["pairs"], np.nanmedian(td["tau_s"], 0))},
           "gamma_array_median": _r(np.nanmedian(cp["gamma_array"])),
           "gamma_array_iqr": _r(np.nanpercentile(cp["gamma_array"], 75)
                                 - np.nanpercentile(cp["gamma_array"], 25))
           if np.isfinite(cp["gamma_array"]).any() else None,
           "figures": {}}
    if g["linear"]:
        b = bearing(td)
        good = (b["confidence"] >= min_conf) & ~b["clipped"] \
            & np.isfinite(b["bearing_deg"])
        doc["bearing"] = {
            "median_deg": _r(np.median(b["bearing_deg"][good]), 1)
            if good.any() else None,
            "iqr_deg": _r(np.percentile(b["bearing_deg"][good], 75)
                          - np.percentile(b["bearing_deg"][good], 25), 1)
            if good.any() else None,
            "confident_fraction": _r(good.mean()),
            "confidence_median": _r(np.median(b["confidence"])),
            "note": "bearing from the array axis, 0-180 deg; the two "
                    "sides of the axis fold together (front-back "
                    "ambiguity) and endfire bearings are low-resolution"}
        doc["figures"]["bearing"] = str(bearing_figure(
            b, out / "array_bearing.png", title=name))
    else:
        doc["bearing"] = None
    doc["figures"]["coherence"] = str(coherence_figure(
        cp, out / "array_coherence.png", title=name))
    (out / "array.json").write_text(json.dumps(doc, indent=2))
    return doc
