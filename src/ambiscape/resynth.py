"""Recreate a session's soundscape from layers of basic synthesis models.

Not a sampler: nothing here plays back recorded audio. The session's
*analysis* is distilled into a compact **recipe** — octave-band background
spectrum, tonal machine lines, event statistics, spatial character — and
the recipe drives four canonical synthesis blocks rendered as a
self-contained Web Audio page (no external resources, one HTML file):

1. *bed* — subtractive synthesis: white noise shaped by octave filters,
2. *machine* — additive synthesis: detuned oscillator pairs with slow AM,
3. *events* — stochastic scheduling: enveloped noise bursts at the
   measured event rate,
4. *space* — panning and a small feedback-delay diffusion.

Written as teaching material: each layer's parameters are visible in the
page, sliders change them live, and the mapping from measured descriptor
to synthesis parameter is spelled out in comments. The recipe degrades
gracefully — module JSONs that were never produced simply leave their
layer at defaults.

Corrected 2026-08-02, after reading the implementation against Andy Farnell's *Designing Sound*
(MIT Press, 2010). Six defects, of which the first two were audible:

1. the bed realised its measured spectrum through ten peaking biquads in SERIES at Q = 1. Modelled
   with the Web Audio biquad formulas against a representative curve, the realised response was out
   by up to 28.9 dB (rms 17.5) and put the top octave 6 dB above the one below it where the
   measurement says 8 dB below. It is now a parallel bandpass bank at Q = 1.414, summed: max error
   3.7 dB, rms 2.5, and the top-octave trend matches. The residual is smooth positive leakage from
   adjacent bands and could be trimmed at build time in `build_recipe`;
2. the space send bus was connected to the master as well as through its own wet path, so everything
   sent to it was heard a second time dry and the bed reached the master three times;
3. every event burst played the noise buffer from sample zero, so all events were the identical
   waveform;
4. the measured `event_median_dur_s` was multiplied by three;
5. event amplitude was fixed, so bursts differed in neither waveform nor level;
6. the tonal pair beat at 1.7 Hz whatever was measured, and every oscillator started at phase zero
   together, so N lines summed as N rather than as sqrt(N).

The event rate slider also capped at 20 per minute in the markup and silently clamped anything
higher that the recording had measured.

What Farnell would still add, and has not been done: a control layer for the bed, since `typical_gain_db`
is computed, shipped in the recipe and never used by the page, and `typ - bg` per band is a measured
modulation depth; a proper grain scheduler with a look-ahead against `ctx.currentTime` rather than
`setTimeout`, degrading to a signal-domain texture above the density where discrete grains stop being
affordable; and a `ConvolverNode` with a synthesised impulse response in place of the two cross-fed
delays, which give about 50 echoes per second against the roughly 1000 that Schroeder suggests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import OCT_CENTERS

EPS = 1e-12


def _load(analysis_dir: Path, name: str) -> dict:
    p = Path(analysis_dir) / name
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def build_recipe(sess, F: dict, analysis_dir) -> dict:
    """Distill analysis outputs into synthesis parameters."""
    analysis_dir = Path(analysis_dir)
    oct_db = 10 * np.log10(np.asarray(F["oct_pow"], np.float64) + EPS)
    bg = np.percentile(oct_db, 10, axis=0)          # background per octave
    typ = np.median(oct_db, axis=0)                 # typical per octave
    bg -= bg.max()
    typ -= typ.max()

    tonality = _load(analysis_dir, "tonality.json")
    tones = [{"freq_hz": float(tr["f_median_hz"]),
              "prominence_db": float(tr.get("prominence_db", 6.0)),
              "minutes": int(tr.get("minutes", 1))}
             for tr in tonality.get("tracks", [])][:6]

    modulation = _load(analysis_dir, "modulation.json")
    micro = modulation.get("scales", {}).get("micro", {})

    summary = _load(analysis_dir, "summary.json")
    spatial = _load(analysis_dir, "spatial.json")
    diffuse = float(np.median(np.asarray(F.get("diffuse", [0.5]),
                                         np.float64)))

    recipe = {
        "session": sess.name,
        "meta": {
            "duration_min": summary.get("duration_min"),
            "laeq_dbfs": summary.get("laeq_dbfs"),
            "note": ("parameters measured by ambiscape; sliders start at "
                     "the measured values"),
        },
        "layers": {
            "bed": {
                "octave_hz": [float(c) for c in OCT_CENTERS],
                "gain_db": [round(float(g), 1) for g in bg],
                "typical_gain_db": [round(float(g), 1) for g in typ],
            },
            "machine": {
                "tones": tones,
                "am_freq_hz": float(micro.get("peak_freq_hz", 0.5)),
                "am_depth": float(micro.get("modulation_depth", 0.3)),
            },
            "events": {
                "per_min": float(summary.get("events_per_min", 0.0)),
                "dur_s": float(summary.get("event_median_dur_s", 0.3)),
                "center_hz": float(summary.get("centroid_median_hz", 800.0)),
                "passbys": len(spatial.get("passbys", [])),
                "passby_dur_s": float(
                    spatial.get("passbys", [{}])[0].get("dur_s", 20.0)
                    if spatial.get("passbys") else 20.0),
            },
            "space": {
                "diffuseness": round(diffuse, 2),
                "azimuth_R": float(spatial.get("azimuth_R_median", 0.8)),
            },
        },
    }
    return recipe


def write_page(recipe: dict, out_path) -> Path:
    """Render the self-contained Web Audio page for a recipe."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = _PAGE.replace("__RECIPE__", json.dumps(recipe, indent=1))
    out_path.write_text(html)
    return out_path


def run_session(sess, F: dict, analysis_dir, out_dir=None) -> dict:
    out_dir = (Path(out_dir) if out_dir
               else Path(analysis_dir).parent / "resynthesis")
    recipe = build_recipe(sess, F, analysis_dir)
    page = write_page(recipe, out_dir / "index.html")
    (out_dir / "recipe.json").write_text(
        json.dumps(recipe, indent=2, default=float))
    return {"out_path": str(page), "layers": list(recipe["layers"])}


_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ambiscape resynthesis</title>
<style>
 :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
 body { max-width: 60rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }
 h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 0; }
 section { border: 1px solid color-mix(in srgb, currentColor 25%, transparent);
           border-radius: .6rem; padding: 1rem; margin: 1rem 0; }
 .row { display: flex; align-items: center; gap: .8rem; flex-wrap: wrap; }
 input[type=range] { flex: 1; min-width: 10rem; }
 button { font: inherit; padding: .35rem .9rem; border-radius: .4rem; }
 small, .how { opacity: .75; } .how { display: block; margin-top: .5rem; }
 #start { font-size: 1.1rem; }
 .val { font-variant-numeric: tabular-nums; min-width: 4.5rem;
        text-align: right; }
</style>
</head>
<body>
<h1>Soundscape resynthesis — <span id="sessname"></span></h1>
<p>A recreation of a real recorded place using only <em>basic synthesis
models</em>. Every starting value below was <strong>measured</strong> from
the recording by <a>ambiscape</a> (see the embedded recipe). Toggle layers
to hear how the scene is assembled; drag sliders to exaggerate or remove
each ingredient.</p>
<p class="row"><button id="start">&#9658; start audio</button>
 <span class="row" style="flex:1">master
 <input type="range" id="master" min="-40" max="0" value="-12" step="1">
 <span class="val" id="masterv">-12 dB</span></span></p>

<section data-layer="bed">
 <h2>1 · Background bed — <small>subtractive synthesis</small></h2>
 <div class="row"><label><input type="checkbox" class="on" checked> on</label>
  gain <input type="range" class="gain" min="-40" max="12" value="0" step="1">
  <span class="val gv">0 dB</span></div>
 <span class="how">White noise pushed through one peaking filter per octave
 band (31.5 Hz … 16 kHz). The filter gains are the recording's measured
 10th-percentile spectrum — the level each band sinks to when nothing is
 happening: Schafer's <em>keynote</em>, as a filter curve.</span>
</section>

<section data-layer="machine">
 <h2>2 · Machine tones — <small>additive synthesis + AM</small></h2>
 <div class="row"><label><input type="checkbox" class="on" checked> on</label>
  gain <input type="range" class="gain" min="-40" max="12" value="0" step="1">
  <span class="val gv">0 dB</span></div>
 <div id="tonelist"><small>no tonal tracks detected in this session</small></div>
 <span class="how">One slightly-detuned oscillator <em>pair</em> per tonal
 line ambiscape tracked (ventilation whine, motor hum …), so the pair
 beats gently like a real machine. A low-frequency oscillator amplitude-
 modulates the whole layer at the measured micro-modulation rate.</span>
</section>

<section data-layer="events">
 <h2>3 · Events — <small>stochastic scheduling + envelopes</small></h2>
 <div class="row"><label><input type="checkbox" class="on" checked> on</label>
  gain <input type="range" class="gain" min="-40" max="12" value="0" step="1">
  <span class="val gv">0 dB</span>
  rate <input type="range" class="rate" min="0" max="20" value="1" step="0.5">
  <span class="val rv"></span></div>
 <span class="how">Noise bursts through a bandpass at the recording's
 spectral centroid, each with a fast attack and the measured median event
 duration as decay, fired by a Poisson clock at the measured events/min.
 If the analysis found pass-bys, an occasional long swell pans across the
 stereo field with a filter sweep.</span>
</section>

<section data-layer="space">
 <h2>4 · Space — <small>panning + feedback-delay diffusion</small></h2>
 <div class="row"><label><input type="checkbox" class="on" checked> on</label>
  diffusion <input type="range" class="wet" min="0" max="100" value="30" step="1">
  <span class="val wv"></span></div>
 <span class="how">Bed and diffusion width follow the measured diffuseness
 &psi;; event panning spread follows the azimuthal concentration R. The
 diffusion is two cross-fed delays — the smallest possible "reverb".</span>
</section>

<script type="application/json" id="recipe">__RECIPE__</script>
<script>
"use strict";
const R = JSON.parse(document.getElementById("recipe").textContent);
document.getElementById("sessname").textContent = R.session;
const L = R.layers;
const $ = (sel, el) => (el || document).querySelector(sel);
const db2lin = db => Math.pow(10, db / 20);

let ctx = null;
const nodes = {};           // per-layer { in (GainNode), on, gain }

function noiseBuffer(seconds) {
  // one reusable buffer of white noise; loops are inaudible at 4 s
  const n = Math.floor(seconds * ctx.sampleRate);
  const buf = ctx.createBuffer(1, n, ctx.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
  return buf;
}

function makeLayer(name, toMaster = true) {
  const g = ctx.createGain();
  // The space layer is a SEND bus: it reaches the master through its own wet path and must not
  // also be connected here. It was, so everything sent to space was heard a second time dry at
  // unity, and the bed reached the master three times over. Toggling the space checkbox zeroed the
  // bus and hid the symptom, which is probably why it survived.
  if (toMaster) g.connect(nodes.master);
  nodes[name] = { in: g, on: true, gain: 1 };
  return g;
}

function startAudio() {
  ctx = new AudioContext();
  nodes.master = ctx.createGain();
  nodes.master.gain.value = db2lin(+$("#master").value);
  nodes.master.connect(ctx.destination);

  // ---- 4 · space: two cross-fed delays as a minimal diffusion network.
  // Feedback < 1 keeps it stable; wet level starts at measured psi.
  const space = makeLayer("space", false);
  const wet = ctx.createGain();
  wet.gain.value = L.space.diffuseness * 0.5;
  const d1 = ctx.createDelay(); d1.delayTime.value = 0.031;
  const d2 = ctx.createDelay(); d2.delayTime.value = 0.047;
  const fb1 = ctx.createGain(); fb1.gain.value = 0.35;
  const fb2 = ctx.createGain(); fb2.gain.value = 0.35;
  const pL = ctx.createStereoPanner(); pL.pan.value = -0.7;
  const pR = ctx.createStereoPanner(); pR.pan.value = 0.7;
  space.connect(d1); space.connect(d2);
  d1.connect(fb1); fb1.connect(d2);   // cross-feed
  d2.connect(fb2); fb2.connect(d1);
  d1.connect(pL); d2.connect(pR);
  pL.connect(wet); pR.connect(wet);
  wet.connect(nodes.master);
  nodes.space.wet = wet;
  const sendToSpace = node => node.connect(space);

  // ---- 1 · bed: white noise -> a PARALLEL bank of octave bandpasses, summed.
  // Gains are the measured 10th-percentile (background) spectrum.
  //
  // This used to be ten `peaking` biquads in SERIES at Q = 1. That does not realise the measured
  // spectrum. A peaking biquad at Q = 1 is about 1.4 octaves wide, so at octave spacing every
  // filter's gain leaks into its neighbours and the dB values accumulate down the chain. Measured
  // against a representative curve the realised response was out by up to 28.9 dB, it exaggerated
  // the overall tilt from 42 to 50 dB, and it put the top octave 6 dB ABOVE the one below it where
  // the measurement says 8 dB below. The page claimed these gains were the recording's spectrum;
  // they were not.
  //
  // A parallel bank is also the right structure for what is actually measured. `oct_pow` is per-band
  // POWER, and incoherent bands power-sum, which is what summing separate filters does. Q = 1.414
  // gives roughly one octave of bandwidth.
  const bed = makeLayer("bed");
  const src = ctx.createBufferSource();
  src.buffer = noiseBuffer(4); src.loop = true;
  const bedSum = ctx.createGain();
  L.bed.octave_hz.forEach((f, i) => {
    const bq = ctx.createBiquadFilter();
    bq.type = "bandpass"; bq.frequency.value = f; bq.Q.value = 1.414;
    const bg = ctx.createGain(); bg.gain.value = db2lin(L.bed.gain_db[i]);
    src.connect(bq); bq.connect(bg); bg.connect(bedSum);
  });
  const bedTrim = ctx.createGain(); bedTrim.gain.value = 0.5;
  bedSum.connect(bedTrim); bedTrim.connect(bed); sendToSpace(bedTrim);
  src.start();

  // ---- 2 · machine: a detuned oscillator pair per tracked tonal line,
  // level from the line's measured prominence, all AM'd by one LFO.
  const mach = makeLayer("machine");
  const am = ctx.createGain(); am.gain.value = 1.0; am.connect(mach);
  const lfo = ctx.createOscillator();
  lfo.frequency.value = Math.max(0.05, L.machine.am_freq_hz);
  const lfoDepth = ctx.createGain();
  lfoDepth.gain.value = Math.min(0.9, L.machine.am_depth) * 0.5;
  lfo.connect(lfoDepth); lfoDepth.connect(am.gain); lfo.start();
  const list = $("#tonelist"); if (L.machine.tones.length) list.textContent = "";
  L.machine.tones.forEach(t => {
    const lvl = ctx.createGain();
    lvl.gain.value = db2lin(-30 + Math.min(24, t.prominence_db));
    // The beat rate scales with the partial rather than being 1.7 Hz for every line regardless of
    // what was measured, and the two oscillators start at independent times so that separate tonal
    // lines sum incoherently. Starting every oscillator at phase 0 simultaneously makes N lines sum
    // as N rather than as sqrt(N).
    const beat = Math.max(0.3, Math.min(4.0, t.freq_hz * 0.004));
    [0, beat].forEach(det => {
      const o = ctx.createOscillator();
      o.frequency.value = t.freq_hz + det;
      o.connect(lvl); o.start(ctx.currentTime + Math.random() * 0.05);
    });
    lvl.connect(am); sendToSpace(lvl);
    const p = document.createElement("small");
    p.textContent = `${Math.round(t.freq_hz)} Hz ` +
      `(${t.prominence_db} dB prominent, ${t.minutes} min) `;
    list.appendChild(p);
  });

  // ---- 3 · events: Poisson-scheduled enveloped noise bursts.
  const ev = makeLayer("events");
  const rateEl = $(".rate", $('[data-layer="events"]'));
  // the slider capped at 20/min in the markup and silently clamped anything the recording measured
  // above that, so the page showed a rate the recording did not have
  rateEl.max = Math.max(20, Math.ceil(L.events.per_min * 2));
  rateEl.value = L.events.per_min;
  const evBuf = noiseBuffer(2);
  function burst() {
    const s = ctx.createBufferSource(); s.buffer = evBuf;
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass"; bp.frequency.value = L.events.center_hz;
    bp.Q.value = 2;
    const env = ctx.createGain(); env.gain.value = 0;
    const pan = ctx.createStereoPanner();
    // panning spread widens as azimuthal concentration R drops
    pan.pan.value = (Math.random() * 2 - 1) * (1.1 - L.space.azimuth_R);
    s.connect(bp); bp.connect(env); env.connect(pan); pan.connect(ev);
    sendToSpace(env);
    const now = ctx.currentTime, dur = L.events.dur_s;
    // Amplitude varies per event. A fixed level made every burst identical in loudness as well as
    // in waveform, which reads as a machine rather than as a room.
    const peak = 0.4 + Math.random() * 0.5;
    env.gain.linearRampToValueAtTime(peak, now + 0.02);      // attack
    env.gain.exponentialRampToValueAtTime(0.001, now + 0.02 + dur);
    // Start at a random offset into the noise buffer. Every burst used to play it from sample zero,
    // so all events were the SAME waveform, and whatever spectral accident sat in the first 300 ms
    // was repeated for the life of the page.
    const off = Math.random() * Math.max(0, evBuf.duration - dur - 0.1);
    // The measured median duration is used as measured. It was multiplied by three here, which
    // silently undid `event_median_dur_s`.
    s.start(now, off); s.stop(now + 0.05 + dur);
  }
  function passby() {
    const s = ctx.createBufferSource(); s.buffer = evBuf; s.loop = true;
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass"; lp.frequency.value = 300; lp.Q.value = 1;
    const env = ctx.createGain(); env.gain.value = 0;
    const pan = ctx.createStereoPanner(); pan.pan.value = -1;
    s.connect(lp); lp.connect(env); env.connect(pan); pan.connect(ev);
    sendToSpace(env);
    const now = ctx.currentTime, dur = L.events.passby_dur_s;
    env.gain.linearRampToValueAtTime(0.5, now + dur * 0.5); // approach
    env.gain.linearRampToValueAtTime(0, now + dur);         // recede
    lp.frequency.linearRampToValueAtTime(1200, now + dur * 0.5);
    lp.frequency.linearRampToValueAtTime(250, now + dur);   // "doppler" dulling
    pan.pan.linearRampToValueAtTime(1, now + dur);          // sweep the field
    s.start(now); s.stop(now + dur + 0.1);
  }
  (function schedule() {
    if (!ctx) return;
    const perMin = +$(".rate", $('[data-layer="events"]')).value;
    $(".rv", $('[data-layer="events"]')).textContent =
      perMin.toFixed(1) + "/min";
    // Poisson process: exponential inter-arrival times
    const wait = perMin > 0 ? -Math.log(1 - Math.random()) / (perMin / 60)
                            : 5;
    setTimeout(() => {
      if (nodes.events.on && perMin > 0) {
        if (L.events.passbys > 0 && Math.random() < 0.1) passby();
        else burst();
      }
      schedule();
    }, Math.min(wait, 30) * 1000);
  })();

  wireUI();
}

function wireUI() {
  document.querySelectorAll("section[data-layer]").forEach(sec => {
    const name = sec.dataset.layer, n = nodes[name];
    const apply = () => {
      n.in.gain.value = n.on ? n.gain : 0;
      if (name === "space" && n.wet)
        n.wet.gain.value = (n.on ? 1 : 0) * (+$(".wet", sec).value / 100);
    };
    $(".on", sec).addEventListener("change", e => {
      n.on = e.target.checked; apply();
    });
    const g = $(".gain", sec);
    if (g) g.addEventListener("input", () => {
      n.gain = db2lin(+g.value);
      $(".gv", sec).textContent = g.value + " dB";
      apply();
    });
    const w = $(".wet", sec);
    if (w) {
      const show = () => $(".wv", sec).textContent = w.value + " %";
      w.value = Math.round(L.space.diffuseness * 50); show();
      w.addEventListener("input", () => { show(); apply(); });
    }
    apply();
  });
  $("#master").addEventListener("input", e => {
    nodes.master.gain.value = db2lin(+e.target.value);
    $("#masterv").textContent = e.target.value + " dB";
  });
}

$("#start").addEventListener("click", () => {
  if (!ctx) { startAudio(); $("#start").textContent = "audio running"; }
});
</script>
</body>
</html>
"""
