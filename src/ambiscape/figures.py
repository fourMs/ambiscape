"""Session figures.

Names follow ambiviz conventions where the plots correspond
(https://github.com/fisheggg/ambiviz): the azimuth-vs-time panel is an
*anglegram* and the polar energy histogram a *directogram*, computed here from
streaming per-second pseudo-intensity features rather than a full AEM, so they
scale to many-hour recordings. For rich spherical maps (AEM) of short
excerpts, export a segment and use ambiviz directly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .analysis import db, detect_events, running_background

INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, SURF = "#e1e0d9", "#fcfcfb"
BLUE, GREEN, MAGENTA, YELLOW = "#2a78d6", "#008300", "#e87ba4", "#eda100"
SEQ = LinearSegmentedColormap.from_list("seqblue", [
    SURF, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
    "#184f95", "#0d366b"])

RC = {
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": SEC, "text.color": INK,
    "xtick.color": MUT, "ytick.color": MUT, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9.5,
}


def _time_axis(ax, t, clock=None):
    span = t[-1] - t[0]
    step = 3600 if span > 5400 else (600 if span > 900 else 120)
    ticks = np.arange(np.ceil(t[0] / step) * step, t[-1], step)
    ax.set_xticks(ticks)
    if clock:
        ax.set_xticklabels([clock(x)[-8:-3] for x in ticks])
    ax.set_xlim(t[0], t[-1])


def _gap_split(t, max_gap=600.0):
    """Split a time axis into contiguous [i0, i1) index ranges at gaps."""
    if len(t) == 0:
        return []
    cuts = np.flatnonzero(np.diff(t) > max_gap) + 1
    edges = [0, *cuts.tolist(), len(t)]
    return list(zip(edges[:-1], edges[1:]))


def overview(F, out_path, title="", clock=None):
    """4-row overview: fast level + background; log spectrogram; anglegram
    (energy-weighted azimuth x time); diffuseness. Takes separated by more
    than 10 minutes get their own column (width ~ duration)."""
    with plt.rc_context(RC):
        segs = _gap_split(F["t"])
        widths = [F["t"][i1 - 1] - F["t"][i0] + 1 for i0, i1 in segs]
        fig, axes = plt.subplots(
            4, len(segs), figsize=(12, 9.5), dpi=130, sharey="row",
            gridspec_kw={"width_ratios": widths, "wspace": 0.04}, squeeze=False)

        tf_all, fast_all = F["t_fast"], F["fast_db"]
        dt = float(np.median(np.diff(tf_all))) if len(tf_all) > 1 else 0.125
        n_events_total = 0
        nonempty = F["logspec"].sum(0) > 0
        Sall = db(F["logspec"][:, nonempty])
        vmax = np.percentile(Sall, 99.5)
        fc = np.sqrt(F["logf"][:-1] * F["logf"][1:])[nonempty]
        p_all = F["rms_w"].astype(np.float64) ** 2

        for col, (i0, i1) in enumerate(segs):
            t = F["t"][i0:i1]
            fm = (tf_all >= t[0]) & (tf_all < t[-1] + 1)
            tf, fast = tf_all[fm], fast_all[fm]
            events, bg = detect_events(fast, dt)
            n_events_total += len(events)

            ax = axes[0][col]
            ax.plot(tf, fast, color=BLUE, lw=0.4, alpha=0.7)
            ax.plot(tf, bg, color=YELLOW, lw=1.4)
            for q, ls in ((90, ":"), (50, "-"), (10, ":")):
                ax.axhline(np.percentile(fast_all, q), color=MUT, lw=0.7, ls=ls)
            ax.set_xlim(t[0], t[-1])

            ax = axes[1][col]
            ax.pcolormesh(t, fc, Sall[i0:i1].T, cmap=SEQ, vmin=vmax - 65,
                          vmax=vmax, shading="auto", rasterized=True)
            ax.set_yscale("log")
            ax.set_ylim(25, 16000)
            ax.grid(False)

            ax = axes[2][col]
            nbins = 36
            nb_t = max(int((t[-1] - t[0]) / 30.0), 1)
            tb = np.linspace(t[0], t[-1] + 1, nb_t + 1)
            H = np.zeros((nbins, nb_t))
            azb = np.linspace(-180, 180, nbins + 1)
            ti = np.clip(np.searchsorted(tb, t) - 1, 0, nb_t - 1)
            ai = np.clip(np.searchsorted(azb, F["az"][i0:i1]) - 1, 0, nbins - 1)
            np.add.at(H, (ai, ti), p_all[i0:i1])
            ax.pcolormesh(tb[:-1], azb[:-1], db(H), cmap=SEQ, shading="auto",
                          rasterized=True, vmin=db(H).max() - 40,
                          vmax=db(H).max())
            ax.set_yticks([-180, -90, 0, 90, 180])
            ax.grid(False)

            ax = axes[3][col]
            d = F["diffuse"][i0:i1]
            ax.plot(t, d, color=GREEN, lw=0.5, alpha=0.55)
            k = min(121, max(3, len(t) // 50) | 1)
            ax.plot(t, np.convolve(d, np.ones(k) / k, "same"),
                    color=GREEN, lw=1.6)
            ax.set_ylim(0, 1)
            for row in range(4):
                _time_axis(axes[row][col], t, clock)
                if row < 3:
                    axes[row][col].tick_params(labelbottom=False)
            if clock and len(segs) > 1:
                axes[0][col].set_title(clock(t[0])[:6], loc="left",
                                       fontsize=8.5, color=SEC)

        axes[0][0].set_ylabel("fast level (dBFS)")
        axes[1][0].set_ylabel("frequency (Hz)")
        axes[2][0].set_ylabel("azimuth (°)\nanglegram")
        axes[3][0].set_ylabel("diffuseness ψ")
        fig.suptitle(f"{title} — level (blue), running background (yellow), "
                     f"L10/L50/L90 (grey); {n_events_total} events",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def ltas_percentiles(F, out_path, title=""):
    """10/50/90th percentile long-term spectra (background vs foreground)."""
    with plt.rc_context(RC):
        nonempty = F["logspec"].sum(0) > 0
        S = db(F["logspec"][:, nonempty])
        fc = np.sqrt(F["logf"][:-1] * F["logf"][1:])[nonempty]
        fig, ax = plt.subplots(figsize=(8, 4), dpi=130)
        p10, p50, p90 = (np.percentile(S, q, axis=0) for q in (10, 50, 90))
        ax.fill_between(fc, p10, p90, color=BLUE, alpha=0.18, lw=0)
        ax.plot(fc, p50, color=BLUE, lw=1.5)
        ax.plot(fc, p10, color=MUT, lw=0.8)
        ax.plot(fc, p90, color=MAGENTA, lw=1.0)
        ax.annotate("90th pct (foreground)", (fc[-30], p90[-30]), color=MAGENTA,
                    fontsize=8, xytext=(0, 6), textcoords="offset points")
        ax.annotate("10th pct (background)", (fc[-30], p10[-30]), color=MUT,
                    fontsize=8, xytext=(0, -12), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (dB)")
        ax.set_title(f"{title} — percentile LTAS", loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def directogram(F, out_path, title=""):
    """Polar azimuth histograms: foreground (loudest 25 %) vs background
    (quietest 25 %) energy-weighted. ambiviz-style directogram."""
    with plt.rc_context(RC):
        p = F["rms_w"].astype(np.float64) ** 2
        fg, bgm = p >= np.percentile(p, 75), p <= np.percentile(p, 25)
        fig, ax = plt.subplots(figsize=(5, 5), dpi=130,
                               subplot_kw={"projection": "polar"})
        bins = np.linspace(-180, 180, 37)
        th = np.radians((bins[:-1] + bins[1:]) / 2)
        for m, c, lab in ((bgm, MUT, "background (quietest 25%)"),
                          (fg, BLUE, "foreground (loudest 25%)")):
            h, _ = np.histogram(F["az"][m], bins=bins, weights=p[m])
            ax.bar(th, h / (h.max() + 1e-20), width=np.radians(10) * 0.92,
                   color=c, alpha=0.65, label=lab)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        ax.set_thetagrids([0, 90, 180, 270],
                          ["front", "left", "rear", "right"], fontsize=8.5)
        ax.set_rticks([])
        ax.set_title(f"{title} — directogram", fontsize=10, pad=14)
        ax.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.12),
                  frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
