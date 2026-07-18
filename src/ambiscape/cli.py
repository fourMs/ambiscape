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
    bn = sub.add_parser("birdnet",
                        help="BirdNET bird-species detections, optionally on "
                             "hi-fi windows only (needs ambiscape[ml])")
    bn.add_argument("folder")
    bn.add_argument("-o", "--out", default=None)
    bn.add_argument("--lat", type=float, default=None)
    bn.add_argument("--lon", type=float, default=None)
    bn.add_argument("--min-conf", type=float, default=0.25)
    bn.add_argument("--hifi-max-diffuse", type=float, default=None,
                    help="skip windows whose median diffuseness exceeds this "
                         "(needs a prior analyze run; e.g. 0.75)")
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
    mu = sub.add_parser("music",
                        help="librosa tempogram + chromagram "
                             "(needs ambiscape[music] and a prior analyze)")
    mu.add_argument("folder")
    mu.add_argument("-o", "--out", default=None)
    mu.add_argument("--t0", type=float, default=0.0)
    mu.add_argument("--dur", type=float, default=None)
    for name, help_ in (("spatial", "direct/diffuse split, pass-by events, "
                                    "azimuth organization timeline"),
                        ("schedule", "match event streams against civic "
                                     "time grids"),
                        ("timbre", "event timbre templates (rise/decay "
                                   "fingerprints, clustered)")):
        sp = sub.add_parser(name, help=help_ + " (needs a prior analyze run)")
        sp.add_argument("folder")
        sp.add_argument("-o", "--out", default=None)
    rs = sub.add_parser("resolve",
                        help="state-resolved descriptors: summarize each "
                             "machine on/off or day/night state separately "
                             "(needs a prior analyze run)")
    rs.add_argument("folder")
    rs.add_argument("-o", "--out", default=None)
    rs.add_argument("--by", choices=("machine", "diel"), default="machine",
                    help="split by machine band on/off (default) or wall-clock "
                         "day/night")
    rs.add_argument("--band", default="250,1000",
                    help="machine band lo,hi Hz for --by machine "
                         "(default 250,1000)")
    rs.add_argument("--night", default="22,6",
                    help="night start,end hour for --by diel (default 22,6)")
    cat = sub.add_parser("catalog",
                         help="aggregate every <session>/analysis/summary.json "
                              "in a corpus into one CSV + Markdown table")
    cat.add_argument("corpus", help="folder containing session subfolders")
    cat.add_argument("-o", "--out", default=None,
                     help="CSV path (default <corpus>/analysis/catalog.csv)")
    cat.add_argument("--sort", default=None,
                     help="descriptor key to rank sessions by (prints ranking)")
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

    if args.cmd == "resolve":
        from .io import open_session
        from .features import load_features
        from . import resolve as rmod
        sess = open_session(args.folder)
        out = Path(args.out) if args.out else Path(args.folder) / "analysis"
        paths = sorted((out / "features").glob("*.npz"))
        if not paths:
            print(f"no cached features in {out} — run 'ambiscape analyze'")
            return 1
        F = load_features(paths)
        if args.by == "machine":
            lo, hi = (float(v) for v in args.band.split(","))
            states = rmod.machine_states(F, band=(lo, hi))
        else:
            nlo, nhi = (int(v) for v in args.night.split(","))
            states = rmod.diel_states(F, sess, night=(nlo, nhi))
        res = rmod.resolve(F, states)
        doc = {"states": {k: {"intervals_s": states[k], **v}
                          for k, v in res.items()}}
        (out / "states.json").write_text(json.dumps(doc, indent=2))
        for label, s in res.items():
            print(f"  {label}: {s['duration_min']} min, Leq {s['leq_dbfs']} "
                  f"dBFS, psi {s['diffuseness_median']}, events/min "
                  f"{s['events_per_min']}, NDSI {s.get('ndsi')}")
        print(f"wrote {out/'states.json'}")
        return 0

    if args.cmd == "catalog":
        from . import catalog as cat_mod
        col = cat_mod.collect(args.corpus)
        if not col:
            print(f"no summary.json found under {args.corpus} — run "
                  "'ambiscape analyze' on the sessions first")
            return 1
        out = Path(args.out) if args.out else \
            Path(args.corpus) / "analysis" / "catalog.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        cat_mod.to_csv(col, out)
        (out.with_suffix(".md")).write_text(cat_mod.to_markdown(col))
        print(f"  {len(col)} sessions -> {out} and {out.with_suffix('.md')}")
        if args.sort:
            for name, val in cat_mod.rank(col, args.sort):
                print(f"    {val:>10.2f}  {name}")
        return 0

    if args.cmd == "birdnet":
        from .io import open_session
        from .ml import birdnet_available, birdnet_session
        if not birdnet_available():
            print("birdnetlib not installed — pip install 'ambiscape[ml]'")
            return 1
        sess = open_session(args.folder)
        out = Path(args.out) if args.out else Path(args.folder) / "analysis"
        F = None
        if args.hifi_max_diffuse is not None:
            from .features import load_features
            paths = sorted((out / "features").glob("*.npz"))
            if not paths:
                print(f"no cached features in {out} — run 'ambiscape analyze'")
                return 1
            F = load_features(paths)
        doc = birdnet_session(sess, F=F, lat=args.lat, lon=args.lon,
                              min_conf=args.min_conf,
                              hifi_max_diffuse=args.hifi_max_diffuse)
        out.mkdir(parents=True, exist_ok=True)
        (out / "birdnet.json").write_text(json.dumps(doc, indent=2))
        print(f"  {doc['n_windows_with_birds']}/{doc['n_windows_analyzed']} "
              f"windows with birds, {doc['n_species']} species")
        for s in doc["species"][:10]:
            print(f"    {s['common_name']} ({s['species']}): "
                  f"{s['n']}×, max conf {s['max_conf']}")
        print(f"wrote {out/'birdnet.json'}")
        return 0

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

    if args.cmd == "music":
        from .io import open_session
        from .music import run_session
        sess = open_session(args.folder)
        out = Path(args.out) if args.out else Path(args.folder) / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        doc = run_session(sess, out, t0=args.t0, dur=args.dur)
        print(f"  global tempo {doc['tempo_bpm_global']} BPM "
              f"(period {doc['tempo_period_s']} s); top pitch classes "
              f"{', '.join(doc['top_pitch_classes'])}")
        print(f"wrote {out/'music.png'} and {out/'music.json'}")
        return 0

    if args.cmd in ("spatial", "schedule", "timbre"):
        from .io import open_session
        sess = open_session(args.folder)
        out = Path(args.out) if args.out else Path(args.folder) / "analysis"
        if not (out / "features").exists():
            print(f"no cached features in {out} — run 'ambiscape analyze' first")
            return 1
        mod = {"spatial": "spatial", "schedule": "schedule",
               "timbre": "timbre"}[args.cmd]
        import importlib
        doc = importlib.import_module(f".{mod}", "ambiscape") \
            .run_session(sess, out)
        if args.cmd == "spatial":
            print(f"  azimuth R median {doc['azimuth_R_median']}, "
                  f"{len(doc['passbys'])} pass-by event(s)")
        elif args.cmd == "schedule":
            for m in doc["events"][:3]:
                print(f"  period {m['period_s']:.0f}s: R={m['R']} "
                      f"phase {m['phase_s']}s over {m['n_cycles']} cycle(s)")
        else:
            print(f"  {doc['n_events_fingerprinted']} events -> "
                  f"{doc['n_classes']} timbre class(es), "
                  f"{doc['n_unclustered']} unclustered")
        print(f"wrote {out}/{args.cmd}.json")
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
    from .ecology import summarize_ecology
    summary.update(summarize_ecology(F))
    from .spatial import summarize_spatial
    summary.update(summarize_spatial(F))
    from .biophony import summarize_biophony
    summary.update(summarize_biophony(F))
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
