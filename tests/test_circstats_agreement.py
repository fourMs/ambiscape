"""ambiscape and micromotion must agree where they overlap on angles.

micromotion owns circular statistics across this family of toolboxes: it
carries the axial tests, the circular-linear correlation and the V-test.
ambiscape keeps a small `circstats` for the time-series end --- `phase_stats`
and `relative_phase` --- and the primitives those need, rather than taking a
dependency for six short functions.

That is a defensible division and it decays silently. On 2026-08-12 the two
Rayleigh implementations were found to disagree on about a fifth of random
cases, because ambiscape used Zar's earlier series expansion and micromotion
uses Wilkie's approximation. Both are published, neither was wrong, and
nothing anywhere said they were supposed to match. These tests say so.

They skip when micromotion is not installed, so ambiscape stays installable on
its own; the division is a claim about the family, not a runtime requirement.
"""
import numpy as np
import pytest

from ambiscape import circstats

mm = pytest.importorskip("micromotion.circular",
                         reason="micromotion not installed; "
                                "the agreement claim cannot be checked")


def _angles(rng, n, kappa):
    """`n` von Mises angles at a random mean and the given concentration."""
    return rng.vonmises(rng.uniform(-np.pi, np.pi), kappa, n)


def test_resultant_length_agrees():
    """The mean resultant is the primitive everything else is built on."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = _angles(rng, int(rng.integers(10, 400)), rng.uniform(0.1, 6))
        _, R = circstats.mean_resultant(a)
        assert R == pytest.approx(mm.circ_mean(a)["R"], abs=1e-12)


def test_rayleigh_p_agrees():
    """The case that was actually wrong, and by how much it was wrong.

    Zar's series and Wilkie's approximation differ most where the test is
    least interesting --- strong concentration, where both are tiny --- but
    they differed on a fifth of random draws, and a p-value that depends on
    which toolbox computed it is not a p-value.
    """
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(10, 400))
        a = _angles(rng, n, rng.uniform(0.1, 6))
        R = float(np.abs(np.sum(np.exp(1j * a))) / n)
        assert circstats.rayleigh_p(R, n) == pytest.approx(
            mm.rayleigh(a)["p"], rel=1e-9, abs=1e-300)


def test_circular_correlation_agrees():
    """Same name in both, and different return types --- float against dict.

    The values agree; the signatures do not, and a caller who swaps the import
    gets an object rather than a number. Left as it is because changing either
    is an API break, and pinned here so at least the arithmetic cannot drift
    on top of the shape difference.
    """
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(20, 300))
        a = _angles(rng, n, rng.uniform(0.5, 4))
        b = _angles(rng, n, rng.uniform(0.5, 4))
        assert circstats.circ_corr(a, b) == pytest.approx(
            mm.circ_corr(a, b)["r"], abs=1e-12)


def test_micromotion_carries_what_ambiscape_does_not():
    """The division is only honest if the fuller theory is really there."""
    for name in ("rayleigh_axial", "axial_dispersion", "circ_corr_linear",
                 "vtest"):
        assert hasattr(mm, name), f"micromotion.circular lacks {name}"
    for name in ("phase_stats", "relative_phase"):
        assert hasattr(circstats, name), f"ambiscape.circstats lacks {name}"
