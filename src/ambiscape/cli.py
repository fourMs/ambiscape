"""Command-line interface.

    ambiscape probe   <session-folder>            # metadata only
    ambiscape analyze <session-folder> [-o DIR]   # features + figures + README
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ambiscape",
                                 description="Long-duration ambisonic soundscape analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="show session metadata")
    p.add_argument("folder")
    a = sub.add_parser("analyze", help="extract features, figures, README")
    a.add_argument("folder")
    a.add_argument("-o", "--out", default=None,
                   help="output dir (default <folder>/analysis)")
    a.add_argument("--notes", default="", help="free-text session notes for README")
    tx = sub.add_parser("taxonomy",
                        help="render Schaeffer map + Schafer timeline from "
                             "<folder>/annotations.json")
    tx.add_argument("folder")
    tx.add_argument("-o", "--out", default=None)
    dr = sub.add_parser("draft",
                        help="pre-fill annotations.draft.json from detected "
                             "states and events (needs a prior analyze run)")
    dr.add_argument("folder")
    dp = sub.add_parser("deposit",
                        help="export non-identifying 1 Hz feature TSVs "
                             "(StillStanding365 schema) to <folder>/deposit/")
    dp.add_argument("folder")
    sg = sub.add_parser("speechgate",
                        help="silero-vad privacy check: fraction of speech "
                             "in WAV file(s) before publishing")
    sg.add_argument("path", help="a WAV file or a folder of WAVs")
    sg.add_argument("--threshold", type=float, default=0.01,
                    help="max allowed speech fraction (default 0.01)")
    rh = sub.add_parser("rhythm",
                        help="strike-level rhythm analysis of quasi-periodic "
                             "pitched sources (needs a prior analyze run)")
    rh.add_argument("folder")
    rh.add_argument("-o", "--out", default=None)
    rh.add_argument("--sources", type=int, default=2,
                    help="max number of sources to track (default 2)")
    rh.add_argument("--stop", type=float, default=None,
                    help="end of the active section in seconds into the "
                         "session (default: auto from band activity)")
    mo = sub.add_parser("modspec",
                        help="multi-scale envelope modulation profile "
                             "(needs a prior analyze run)")
    mo.add_argument("folder")
    mo.add_argument("-o", "--out", default=None)
    to = sub.add_parser("tonality",
                        help="tonal tracks, harmonicity, pitch-class profile "
                             "(needs a prior analyze run)")
    to.add_argument("folder")
    to.add_argument("-o", "--out", default=None)
    iso_p = sub.add_parser("iso",
                           help="ISO 12913-3 psychoacoustic indicators "
                                "(MoSQITo) on representative segments")
    iso_p.add_argument("folder")
    iso_p.add_argument("--dur", type=float, default=30.0,
                       help="seconds per segment (default 30)")
    iso_p.add_argument("--offset", type=float, default=None,
                       help="dBFS->dB SPL offset override")
    args = ap.parse_args(argv)

    if args.cmd == "deposit":
        from .deposit import export_session
        outs = export_session(args.folder)
        if not outs:
            print("no cached features — run 'ambiscape analyze' first")
            return 1
        for o in outs:
            print(f"wrote {o}")
        return 0

    if args.cmd == "speechgate":
        from .ml import speech_gate
        p = Path(args.path)
        files = sorted(p.glob("*.wav")) + sorted(p.glob("*.WAV")) \
            if p.is_dir() else [p]
        ok = True
        for f in files:
            r = speech_gate(f, threshold=args.threshold)
            verdict = "PASS" if r["passes"] else "FAIL"
            ok &= r["passes"]
            extra = ("" if r["passes"] else
                     f" (first speech at {r['first_speech_at_s']}s)")
            print(f"  {verdict}  {f.name}: {r['speech_fraction']*100:.2f}% "
                  f"speech, {r['n_speech_segments']} segment(s){extra}")
        return 0 if ok else 2

    if args.cmd == "iso":
        from .features import load_features
        from .io import open_session
        from . import iso as iso_mod
        sess = open_session(args.folder)
        fdir = Path(args.folder) / "analysis" / "features"
        paths = sorted(fdir.glob("*.npz"))
        if not paths:
            print(f"no cached features in {fdir} — run 'ambiscape analyze' first")
            return 1
        F = load_features(paths)
        res = iso_mod.segment_indicators(sess, F, args.folder,
                                         dur=args.dur, offset=args.offset)
        out = Path(args.folder) / "analysis" / "iso_indicators.json"
        out.write_text(json.dumps(res, indent=2))
        if not res["calibrated"]:
            print("WARNING:", res["warning"])
        for kind, seg in res["segments"].items():
            print(f"  {kind} @ {seg['t0']}: N5 {seg['N5_sone_max_ear']} sone "
                  f"(max ear), sharpness {seg['left']['sharpness_median_acum']}"
                  f"/{seg['right']['sharpness_median_acum']} acum "
                  f"[{seg['binaural_method']}]")
        print(f"wrote {out}")
        return 0

    if args.cmd == "draft":
        from .features import load_features
        from .io import open_session
        from .draft import draft_annotations
        fdir = Path(args.folder) / "analysis" / "features"
        paths = sorted(fdir.glob("*.npz"))
        if not paths:
            print(f"no cached features in {fdir} — run 'ambiscape analyze' first")
            return 1
        F = load_features(paths)
        try:
            sess = open_session(args.folder)
        except (FileNotFoundError, ValueError):
            sess = None
        out = draft_annotations(F, args.folder, session=sess)
        doc = json.loads(out.read_text())
        n_obj = len(doc["objects"])
        n_tag = sum(1 for o in doc["objects"] for h in o.get("_hints", [{}])
                    if "tags" in h) + sum(1 for o in doc["objects"]
                                          if "_tags" in o)
        print(f"wrote {out} ({n_obj} draft objects, {n_tag} PANNs-tagged "
              f"windows) — edit, save as annotations.json, then run "
              f"'ambiscape taxonomy'")
        return 0

    if args.cmd == "rhythm":
        from .io import open_session
        from .rhythm import run_session
        sess = open_session(args.folder)
        out = Path(args.out) if args.out else Path(args.folder) / "analysis"
        if not (out / "features").exists():
            print(f"no cached features in {out} — run 'ambiscape analyze' first")
            return 1
        print(f"rhythm analysis: {sess.name}")
        summary = run_session(sess, out, n_sources=args.sources,
                              t_stop=args.stop)
        for src in summary["sources"]:
            print(f"  source {src['name']}: period {src['period_s']} s, "
                  f"{src['n_strikes']} strikes, az {src['azimuth_deg']} deg, "
                  f"partials {src['partials_hz'][:5]}...")
        print(f"wrote {out/'rhythm_overview.png'} and {out/'rhythm.json'}")
        return 0

    if args.cmd in ("modspec", "tonality"):
        from .io import open_session
        sess = open_session(args.folder)
        out = Path(args.out) if args.out else Path(args.folder) / "analysis"
        if not (out / "features").exists():
            print(f"no cached features in {out} — run 'ambiscape analyze' first")
            return 1
        if args.cmd == "modspec":
            from .modulation import run_session
            prof = run_session(sess, out)
            for scale, st in prof["scales"].items():
                print(f"  {scale}: peak {st['peak_freq_hz']} Hz "
                      f"(period {st['peak_period_s']} s, "
                      f"+{st['peak_prominence_db']} dB), "
                      f"depth {st['modulation_depth']}")
            print(f"wrote {out/'modulation_profile.png'} and "
                  f"{out/'modulation.json'}")
        else:
            from .tonality import run_session
            doc = run_session(sess, out)
            print(f"  {len(doc['tracks'])} tonal tracks; tonalness median "
                  f"{doc['tonalness_median']}, inharmonicity median "
                  f"{doc['inharmonicity_median']}, top pitch classes "
                  f"{', '.join(doc['top_pitch_classes'])}")
            print(f"wrote {out/'tonality.png'} and {out/'tonality.json'}")
        return 0

    if args.cmd == "taxonomy":
        from .taxonomy import render
        paths = render(args.folder, out_dir=args.out)
        for p in paths:
            print(f"wrote {p}")
        return 0

    from .io import open_session
    sess = open_session(args.folder)

    if args.cmd == "probe":
        print(f"session {sess.name}: {len(sess.takes)} take(s), "
              f"{sess.duration/60:.1f} min total")
        for tk in sess.takes:
            print(f"  {tk.path.name}: {tk.date} {tk.clock}, "
                  f"{tk.duration/60:.1f} min, {tk.channels}ch @{tk.samplerate}")
        return 0

    from . import analysis, features, figures, report
    out = Path(args.out) if args.out else sess.folder / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    print(f"analyzing {sess.name} ({sess.duration/60:.1f} min)")
    paths = features.extract_session(sess, out / "features")
    F = features.load_features(paths)
    summary = analysis.summarize(F)
    from .background import summarize_foreground
    summary.update(summarize_foreground(F))
    from .iso import load_calibration, apply_calibration
    cal = load_calibration(sess.folder)
    if cal and "dbfs_to_dbspl" in cal:
        summary = apply_calibration(summary, cal)
    figures.overview(F, out / "overview.png", title=sess.name, clock=sess.clock)
    figures.ltas_percentiles(F, out / "ltas_percentiles.png", title=sess.name)
    figures.directogram(F, out / "directogram.png", title=sess.name)
    report.write_readme(sess, summary, out, notes=args.notes)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"wrote {out} and {sess.folder/'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
