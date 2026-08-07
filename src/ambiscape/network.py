"""Multi-recorder acoustic network: one building heard from many rooms.

Ambisonics puts several capsules at one point and asks *from which
direction*; an acoustic network puts one recorder in each of several rooms
of a building on a common clock (the SINS deployment style) and asks
*through which fabric*: how strongly, and with what delay, does activity in
one room appear in the others. The rooms become nodes, the walls, doors and
corridors become edges, and the building reads as a graph whose shape
changes over the day — a closed door thins an edge, a shared ventilation
run thickens one, and the room everything couples to is the acoustic hub.

Everything works from the cached 8 Hz fast A-weighted level streams of a
prior ``analyze`` run on each node session; no audio is reopened.

- ``load_network`` --- every analysed node session under one folder;
- ``node_grid`` --- all fast-level streams on one uniform clock grid;
- ``pairwise_coupling`` --- windowed, lag-searched cross-correlation of the
  level envelopes: per-window coupling (adjacency) + lag matrices;
- ``graph_measures`` --- numpy-only graph readings per window: node
  strength (the hub measure), edge density, transitivity;
- ``hourly_measures`` --- the same resolved by hour of day;
- ``network_figure`` --- house graphs at representative hours (node size =
  strength, edge width = coupling, arrows = lag direction) over a
  density-of-the-day timeline;
- ``network_summary_keys`` --- the ``net_`` rows folded into
  ``summary.json`` for the catalogue;
- ``run_network`` --- orchestrate the above into network.json + network.png.

Times follow the feature axis: seconds since midnight of day 0 (nodes dated
on later days are shifted by whole days onto the first node's axis). Lags
are antisymmetric with a fixed sign convention: ``lag_s[i, j] > 0`` means
node ``i`` leads — sound appears at ``i`` first and at ``j`` roughly
``lag_s[i, j]`` seconds later.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from .compare import load_comparison
from .features import FAST

EPS = 1e-12


def _nanmed(a: np.ndarray, axis=0) -> np.ndarray:
    """nanmedian without the all-NaN-slice warning (the matrix diagonal)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis)

# ---------------------------------------------------------------- loading


def load_network(folder: str | Path) -> list[dict]:
    """Analysed node sessions: every subfolder of ``folder`` with features.

    A node is any direct subfolder holding an ``analysis/features`` cache
    (one recorder, one room); nodes are returned sorted by name in the
    :func:`ambiscape.compare.load_comparison` session format. Fewer than
    two such subfolders raise — run ``ambiscape analyze`` in each first.
    """
    folder = Path(folder)
    subs = sorted(p for p in folder.iterdir() if p.is_dir()
                  and list((p / "analysis" / "features").glob("*.npz")))
    if len(subs) < 2:
        raise FileNotFoundError(
            f"need at least two analysed node sessions under {folder} — "
            "run 'ambiscape analyze' in each node folder first")
    return load_comparison(subs)


def node_grid(nodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Fast A-weighted levels of every node on one uniform clock grid.

    Returns ``(t, X)``: ``t`` in seconds since midnight of day 0 at the
    fast rate (8 Hz), ``X`` of shape ``(n_nodes, len(t))`` in dBFS with
    NaN wherever a node has no coverage (before its first take, between
    takes, after its last). Nodes whose resolved date differs from the
    first node's are shifted by whole days onto the same axis, so a
    deployment logged across midnight still lines up sample by sample.
    """
    dt = float(FAST)
    d0 = nodes[0]["date"]
    shifts = [86400.0 * (s["date"].toordinal() - d0.toordinal())
              if d0 is not None and s["date"] is not None else 0.0
              for s in nodes]
    t0 = min(s["F"]["t_fast"][0] + sh for s, sh in zip(nodes, shifts))
    t1 = max(s["F"]["t_fast"][-1] + sh for s, sh in zip(nodes, shifts))
    n = int(round((t1 - t0) / dt)) + 1
    t = t0 + dt * np.arange(n)
    X = np.full((len(nodes), n), np.nan, np.float32)
    for row, (s, sh) in enumerate(zip(nodes, shifts)):
        idx = np.round((s["F"]["t_fast"] + sh - t0) / dt).astype(int)
        ok = (idx >= 0) & (idx < n)
        X[row, idx[ok]] = s["F"]["fast_dba"][ok]
    return t, X


# ---------------------------------------------------------------- coupling


def pairwise_coupling(t: np.ndarray, X: np.ndarray, win_s: float = 120.0,
                      max_lag_s: float = 4.0,
                      min_coverage: float = 0.9) -> dict:
    """Windowed cross-correlation of level envelopes with lag search.

    The grid is cut into non-overlapping ``win_s`` windows; in each, every
    node pair's mean- and trend-removed dB envelopes are cross-correlated
    over lags of ±``max_lag_s`` and the peak is kept. Returns
    ``{"win_t", "coupling", "lag_s"}``: window centres (s), the peak
    normalised correlation per window and pair (symmetric, NaN diagonal —
    the per-window adjacency), and the lag at that peak refined by
    parabolic interpolation (antisymmetric; ``lag_s[w, i, j] > 0`` means
    node ``i`` leads node ``j``). A pair is NaN in any window where either
    node covers less than ``min_coverage`` of it or holds a constant level.
    """
    dt = float(t[1] - t[0])
    L = int(round(win_s / dt))
    K = int(round(max_lag_s / dt))
    if L <= 2 * K:
        raise ValueError("window shorter than the lag search range")
    nn = X.shape[0]
    nwin = X.shape[1] // L
    win_t = float(t[0]) + win_s * (np.arange(nwin) + 0.5)
    coupling = np.full((nwin, nn, nn), np.nan)
    lag = np.full((nwin, nn, nn), np.nan)
    ramp = np.arange(L) - (L - 1) / 2
    ramp_ss = float((ramp ** 2).sum())
    counts = L - np.abs(np.arange(-K, K + 1))
    for w in range(nwin):
        seg = X[:, w * L:(w + 1) * L].astype(float)
        pre = []
        for i in range(nn):
            x, m = seg[i], np.isfinite(seg[i])
            if m.mean() < min_coverage:
                pre.append(None)
                continue
            x = np.where(m, x, x[m].mean()) - x[m].mean()
            x = x - ramp * float((x * ramp).sum() / ramp_ss)   # detrend
            sd = float(np.sqrt((x ** 2).mean()))
            pre.append((x, sd) if sd > 1e-6 else None)
        for i in range(nn):
            for j in range(i + 1, nn):
                if pre[i] is None or pre[j] is None:
                    continue
                (xi, si), (xj, sj) = pre[i], pre[j]
                cc = np.correlate(xi, xj, "full")[L - 1 - K:L + K]
                r = cc / (counts * si * sj)
                p = int(np.argmax(r))
                k = float(p - K)
                if 0 < p < 2 * K:      # parabolic sub-sample refinement
                    den = r[p - 1] - 2 * r[p] + r[p + 1]
                    if den < 0:
                        k += 0.5 * float(r[p - 1] - r[p + 1]) / float(den)
                coupling[w, i, j] = coupling[w, j, i] = min(float(r[p]), 1.0)
                # peak at k = -s when x_j trails x_i by s samples
                lag[w, i, j] = -k * dt
                lag[w, j, i] = k * dt
    return {"win_t": win_t, "coupling": coupling, "lag_s": lag}


# ---------------------------------------------------------------- graphs


def graph_measures(coupling: np.ndarray, threshold: float = 0.35) -> dict:
    """Per-window graph measures from the coupling stack (numpy only).

    Returns ``{"strength" (nwin, n), "density" (nwin,), "transitivity"
    (nwin,)}``. Strength is a node's summed coupling to all others (its
    weighted degree — the acoustic-hub reading); density the fraction of
    pairs whose coupling reaches ``threshold``; transitivity the
    closed-triplet ratio of the thresholded graph (NaN where no node has
    two edges). Windows without any finite pair are NaN throughout.
    """
    C = np.asarray(coupling, float)
    n = C.shape[1]
    valid = np.isfinite(C).any((1, 2))
    strength = np.where(np.isfinite(C), C, 0.0).sum(2)
    A = (np.nan_to_num(C, nan=-1.0) >= threshold).astype(float)
    density = A.sum((1, 2)) / (n * (n - 1))
    deg = A.sum(2)
    triplets = (deg * (deg - 1)).sum(1) / 2
    closed = np.einsum("wij,wjk,wki->w", A, A, A) / 2      # 3 × triangles
    trans = np.where(triplets > 0, closed / np.maximum(triplets, 1), np.nan)
    strength[~valid] = np.nan
    density[~valid] = np.nan
    trans[~valid] = np.nan
    return {"strength": strength, "density": density, "transitivity": trans}


def hourly_measures(win_t: np.ndarray, coupling: np.ndarray,
                    threshold: float = 0.35) -> dict:
    """Graph measures resolved by hour of day.

    Returns ``{"hours", "density", "strength", "hub", "n_windows"}``:
    wall-clock hours (0–23, ascending) that hold at least one valid
    window, the median density over each hour's windows, the median
    strength per node (rows follow ``hours``), the index of the strongest
    node per hour, and the window count per hour. Days repeat onto the
    same 24 hours, so a week-long deployment yields one composite day.
    """
    gm = graph_measures(coupling, threshold)
    hod = (win_t // 3600).astype(int) % 24
    valid = np.isfinite(gm["density"])
    hours = sorted(set(hod[valid].tolist()))
    dens, stre, cnt = [], [], []
    for h in hours:
        m = valid & (hod == h)
        dens.append(float(np.median(gm["density"][m])))
        stre.append(np.median(gm["strength"][m], 0))
        cnt.append(int(m.sum()))
    stre = np.array(stre) if stre else np.zeros((0, coupling.shape[1]))
    return {"hours": np.array(hours, int), "density": np.array(dens),
            "strength": stre,
            "hub": stre.argmax(1) if len(stre) else np.array([], int),
            "n_windows": np.array(cnt, int)}


def representative_hours(hourly: dict, n: int = 3) -> list[int]:
    """Up to ``n`` hours spanning the density range: quietest of the day,
    the median hour, and the busiest — the graph's states worth drawing."""
    hs, d = hourly["hours"], hourly["density"]
    if len(hs) <= n:
        return [int(h) for h in hs]
    order = np.argsort(d, kind="stable")
    picks = [order[0], order[len(order) // 2], order[-1]]
    out = []
    for p in picks:
        if int(hs[p]) not in out:
            out.append(int(hs[p]))
    return sorted(out)


# ---------------------------------------------------------------- figure


def network_figure(names: list[str], res: dict, out_path: str | Path,
                   threshold: float = 0.35, hours: list[int] | None = None,
                   min_lag_s: float = 0.1) -> Path:
    """House graphs at representative hours + a density-of-the-day timeline.

    Top row: one graph per hour — nodes on a circle, node size = median
    strength, edge width = median coupling (drawn from ``threshold`` up),
    an arrowhead pointing from the leading room to the lagging one
    wherever the median lag exceeds ``min_lag_s`` (labelled in seconds).
    Bottom: per-window density dots with an hourly median step, the drawn
    hours shaded.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hourly = hourly_measures(res["win_t"], res["coupling"], threshold)
    if hours is None:
        hours = representative_hours(hourly)
    hours = hours or [int(res["win_t"][0] // 3600) % 24]
    n = len(names)
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    pos = np.stack([np.cos(ang), np.sin(ang)], 1)
    hod = (res["win_t"] // 3600).astype(int) % 24
    ncol = max(len(hours), 2)
    fig = plt.figure(figsize=(3.9 * ncol, 6.2))
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.9, 1], hspace=0.25)
    for col, h in enumerate(hours):
        ax = fig.add_subplot(gs[0, col])
        m = hod == h
        C = _nanmed(res["coupling"][m]) if m.any() else \
            np.full((n, n), np.nan)
        Lg = _nanmed(res["lag_s"][m]) if m.any() else C
        for i in range(n):
            for j in range(i + 1, n):
                c = C[i, j]
                if not np.isfinite(c) or c < threshold:
                    continue
                ax.plot(pos[[i, j], 0], pos[[i, j], 1], color="0.6",
                        lw=0.5 + 5 * c, alpha=0.8, zorder=1,
                        solid_capstyle="round")
                if abs(Lg[i, j]) >= min_lag_s:
                    lead, trail = (i, j) if Lg[i, j] > 0 else (j, i)
                    mid = (pos[lead] + pos[trail]) / 2
                    d = pos[trail] - pos[lead]
                    d = d / (np.hypot(*d) + EPS)
                    ax.annotate("", xy=mid + 0.14 * d, xytext=mid - 0.14 * d,
                                arrowprops={"arrowstyle": "-|>",
                                            "color": "#1a4f8f", "lw": 1.4},
                                zorder=2)
                    ax.text(*(mid + 0.17 * np.array([-d[1], d[0]])),
                            f"{abs(Lg[i, j]):.2f} s", fontsize=7,
                            ha="center", va="center", color="#1a4f8f")
        strength = np.where(np.isfinite(C), C, 0.0).sum(1)
        ax.scatter(pos[:, 0], pos[:, 1],
                   s=200 + 900 * strength / (strength.max() + EPS),
                   c="#2a78d6", alpha=0.9, zorder=3, edgecolors="white")
        for i, name in enumerate(names):
            ax.annotate(name, pos[i] * 1.3, ha="center", va="center",
                        fontsize=8)
        ax.set_title(f"{h:02d}:00–{h + 1:02d}:00 ({int(m.sum())} windows)",
                     fontsize=9)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
    gm = graph_measures(res["coupling"], threshold)
    axd = fig.add_subplot(gs[1, :])
    th = res["win_t"] / 3600
    axd.plot(th, gm["density"], ".", ms=3, color="0.65", label="per window")
    bins = np.unique((res["win_t"] // 3600).astype(int))
    med = [_nanmed(gm["density"][(res["win_t"] // 3600).astype(int) == b])
           for b in bins]
    axd.step(bins, med, where="post", lw=1.5, color="#2a78d6",
             label="hourly median")
    for b in bins:
        if int(b) % 24 in hours:
            axd.axvspan(b, b + 1, color="#2a78d6", alpha=0.08, lw=0)
    axd.set_xlabel("clock (h; > 24 = day 2); shaded = hours drawn above")
    axd.set_ylabel("edge density")
    axd.set_ylim(-0.05, 1.05)
    axd.grid(alpha=0.25, lw=0.5)
    axd.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Acoustic network — node size = strength, edge width = "
                 "coupling, arrows lead → lag", fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------- runner


def network_summary_keys(doc: dict) -> dict:
    """The ``net_`` rows folded into ``summary.json`` for the catalogue."""
    keys = {"net_n_nodes": len(doc["nodes"]),
            "net_density_median": doc["density_median"],
            "net_transitivity_median": doc["transitivity_median"],
            "net_hub_node": doc["hub_node"],
            "net_hub_strength": doc["strength_median"].get(doc["hub_node"])
            if doc["hub_node"] else None}
    if doc.get("strongest_pair"):
        sp = doc["strongest_pair"]
        keys["net_max_coupling"] = sp["coupling"]
        keys["net_max_pair"] = "→".join(sp["nodes"])
        keys["net_max_lag_s"] = sp["lag_s"]
    return keys


def _r(v, nd: int = 3):
    v = float(v)
    return round(v, nd) if np.isfinite(v) else None


def run_network(folder: str | Path, out_dir: str | Path | None = None,
                win_s: float = 120.0, max_lag_s: float = 4.0,
                threshold: float = 0.35, min_coverage: float = 0.9) -> dict:
    """Analyse the acoustic network of one building; write JSON + figure.

    ``folder`` holds one analysed session per recorder (one room each, on
    a common clock). Output goes to ``out_dir`` (default
    ``<folder>/analysis``): ``network.json`` (median coupling and lag
    matrices, per-node strength, hub, density and transitivity, hourly
    breakdown), ``network.png`` (house graphs + density timeline), and
    ``net_`` keys folded into ``summary.json`` there (created if absent),
    so the building joins the catalogue as one row. Returns the
    network.json document.
    """
    folder = Path(folder)
    out = Path(out_dir) if out_dir else folder / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    nodes = load_network(folder)
    names = [s["name"] for s in nodes]
    t, X = node_grid(nodes)
    res = pairwise_coupling(t, X, win_s=win_s, max_lag_s=max_lag_s,
                            min_coverage=min_coverage)
    gm = graph_measures(res["coupling"], threshold)
    n_valid = int(np.isfinite(gm["density"]).sum())
    if n_valid == 0:
        raise ValueError("no window with two overlapping nodes — check the "
                         "recorders share a clock and a time span")
    hourly = hourly_measures(res["win_t"], res["coupling"], threshold)
    C_med = _nanmed(res["coupling"])
    L_med = _nanmed(res["lag_s"])
    strength_med = _nanmed(gm["strength"])
    hub = int(np.nanargmax(strength_med))
    iu = np.triu_indices(len(names), 1)
    strongest = None
    if np.isfinite(C_med[iu]).any():
        b = int(np.nanargmax(C_med[iu]))
        i, j = int(iu[0][b]), int(iu[1][b])
        if np.isfinite(L_med[i, j]) and L_med[i, j] < 0:
            i, j = j, i                       # order leader first
        strongest = {"nodes": [names[i], names[j]],
                     "coupling": _r(C_med[i, j]),
                     "lag_s": _r(L_med[i, j])}
    doc = {"nodes": names,
           "params": {"win_s": win_s, "max_lag_s": max_lag_s,
                      "threshold": threshold,
                      "rate_hz": round(1.0 / float(FAST), 3)},
           "n_windows": n_valid,
           "coupling_median": [[_r(v) for v in row] for row in C_med],
           "lag_median_s": [[_r(v) for v in row] for row in L_med],
           "strength_median": {nm: _r(v)
                               for nm, v in zip(names, strength_med)},
           "hub_node": names[hub],
           "density_median": _r(_nanmed(gm["density"])),
           "transitivity_median": _r(_nanmed(gm["transitivity"])),
           "hourly": {f"{int(h):02d}": {
               "density_median": _r(d),
               "hub_node": names[int(hb)],
               "n_windows": int(c)}
               for h, d, hb, c in zip(hourly["hours"], hourly["density"],
                                      hourly["hub"], hourly["n_windows"])},
           "strongest_pair": strongest,
           "figures": {}}
    doc["figures"]["network"] = str(
        network_figure(names, res, out / "network.png", threshold=threshold))
    (out / "network.json").write_text(json.dumps(doc, indent=2))
    sp = out / "summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    summary.update(network_summary_keys(doc))
    sp.write_text(json.dumps(summary, indent=2))
    return doc
