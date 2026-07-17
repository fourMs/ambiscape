# Deposit export — publishing without publishing audio

Raw recordings from private spaces often cannot be shared; their acoustic
*envelope* can. `ambiscape deposit` writes per-take TSVs of 1 Hz features:

```
Time    level_dbfs    centroid_hz    low_frac    high_frac
0       -39.8         881            0.786       0.189
1       -44.9         135            0.966       0.007
```

A 1 Hz loudness/spectral envelope is far below speech timescales and
carries no intelligible content, so these files are safe for open deposits
(Zenodo and similar) where the WAVs are withheld.

The schema is that of the **StillStanding365** deposit (365 daily
standstill sessions, fourMs/RITMO), making corpora that use it directly
poolable. Method deltas against that deposit's original extractor are
documented in the `deposit` module (W-channel at native rate vs an 8 kHz
four-channel downmix; power vs magnitude band fractions): trends and
dynamics are directly comparable, absolute fraction values differ slightly
by construction.

!!! warning "Directional data and channel conventions"
    When depositing *directional* products, record which B-format
    convention the source files used (see
    [Sessions & conventions](sessions.md)). A convention mismatch produces
    azimuth distributions collapsed onto one axis — an artifact that
    survives into downstream correlations and is invisible unless you know
    to look for it.

What still requires raw audio: fast-level descriptors (Leq/LAeq), events at
the 0.25 s criterion, diffuseness, elevation, spectra beyond the centroid,
psychoacoustic indicators. Plan deposits accordingly: features for
everyone, raw audio under controlled access where the ethics allow.
