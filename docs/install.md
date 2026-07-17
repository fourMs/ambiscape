# Install

```bash
pip install ambiscape
```

Core dependencies are numpy, scipy, soundfile, and matplotlib. Optional
extras enable the heavier subsystems:

| Extra | Enables | Pulls in |
|---|---|---|
| `pip install "ambiscape[iso]"` | `ambiscape iso` — ISO 532-1 loudness, sharpness, roughness | [MoSQITo](https://github.com/Eomys/MoSQITo) |
| `pip install "ambiscape[ml]"` | PANNs tag suggestions in drafts; `ambiscape speechgate` | torch (CPU is fine), panns-inference, silero-vad |
| `pip install "ambiscape[viz]"` | HRIR binaural rendering; AEM visuals on excerpts | [ambiviz](https://github.com/fisheggg/ambiviz) |

Everything degrades gracefully: without an extra installed, the
corresponding feature is skipped or falls back (e.g., binaural rendering
falls back to a documented cardioid pair).

For development:

```bash
git clone https://github.com/fourMs/ambiscape
cd ambiscape
pip install -e ".[dev]"
```

!!! note "First PANNs run"
    `panns-inference` downloads the CNN14 checkpoint (~300 MB) to
    `~/panns_data/` on first use.
