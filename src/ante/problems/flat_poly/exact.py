"""Independent scoring for EinsteinArena `flat-polynomials` (degree 69, +-1 coefficients, minimize).

Score = max_{|z|=1} |p(z)| / sqrt(71), where the official verifier samples the unit circle at
1,000,000 points `linspace(0, 2*pi, 1_000_000)` (endpoint included).

Three independent quantities are provided:

- `grid_score`: our own evaluation on exactly the official sample points (what the arena will report,
  up to float rounding in the last digits).
- `sup_score`: the true continuous supremum, located by a dense FFT scan plus Newton refinement of
  every local maximum of |p|^2. This is >= the sampled value; the gap is the amount by which the
  official score flatters a sequence.
- `sup_upper_bound`: a rigorous upper bound on the continuous supremum from a dense grid and the
  Bernstein inequality, so a claimed record never rests on sampling luck.
"""

from __future__ import annotations

import math

import numpy as np

N = 70
NORM = math.sqrt(N + 1)  # the arena divides by sqrt(len(coefficients) + 1) = sqrt(71)
OFFICIAL_POINTS = 1_000_000


def check_coefficients(coefficients) -> np.ndarray:
    c = np.asarray(coefficients)
    if c.shape != (N,):
        raise ValueError(f"expected {N} coefficients, got shape {c.shape}")
    if not np.all((c == 1) | (c == -1)):
        raise ValueError("all coefficients must be +1 or -1")
    return c.astype(np.float64)


def _abs_on_angles(c: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # |sum_k c_k e^{i k theta}|; coefficient order does not matter for the modulus on the unit circle
    # up to reversal (which conjugates), so this matches np.poly1d's highest-power-first convention.
    k = np.arange(N)
    out = np.empty(theta.shape[0])
    step = 20_000
    for s in range(0, theta.shape[0], step):
        t = theta[s : s + step]
        out[s : s + step] = np.abs(np.exp(1j * np.outer(t, k)) @ c)
    return out


def grid_score(coefficients) -> float:
    """Score on the official 1,000,000-point grid (independent re-implementation)."""
    c = check_coefficients(coefficients)
    theta = np.linspace(0.0, 2.0 * np.pi, OFFICIAL_POINTS)
    return float(np.max(_abs_on_angles(c, theta)) / NORM)


def _power_and_derivs(c: np.ndarray, theta: np.ndarray):
    """|p|^2 and its first two derivatives in theta, via aperiodic autocorrelations."""
    r = np.correlate(c, c, mode="full")[N - 1 :]  # r[k] = sum_j c_j c_{j+k}, k = 0..N-1
    k = np.arange(1, N)
    ang = np.outer(theta, k)
    f = r[0] + 2.0 * np.cos(ang) @ r[1:]
    f1 = -2.0 * np.sin(ang) @ (r[1:] * k)
    f2 = -2.0 * np.cos(ang) @ (r[1:] * k * k)
    return f, f1, f2


def sup_score(coefficients, fft_points: int = 1 << 14) -> float:
    """Continuous sup-norm score: dense scan, then Newton on every near-maximal local maximum."""
    c = check_coefficients(coefficients)
    spec = np.abs(np.fft.fft(c, fft_points)) ** 2
    m = spec.max()
    idx = np.nonzero((spec >= np.roll(spec, 1)) & (spec >= np.roll(spec, -1)) & (spec >= 0.9 * m))[0]
    theta = 2.0 * np.pi * idx / fft_points
    for _ in range(8):
        f, f1, f2 = _power_and_derivs(c, theta)
        ok = f2 < 0
        step = np.where(ok, -f1 / np.where(ok, f2, 1.0), 0.0)
        theta = theta + np.clip(step, -np.pi / fft_points, np.pi / fft_points)
    f, _, _ = _power_and_derivs(c, theta)
    return float(math.sqrt(max(float(f.max()), float(m))) / NORM)


def sup_upper_bound(coefficients, fft_points: int = 1 << 18) -> float:
    """Rigorous upper bound on the continuous score.

    q(theta) = |p(e^{i theta})|^2 is a real trigonometric polynomial of degree N-1, so by Bernstein
    |q''| <= (N-1)^2 max q. Between grid points spaced h apart, a maximum of q exceeds the nearest
    grid value by at most (h/2)^2/2 * (N-1)^2 * max q, hence max q <= grid_max / (1 - (h (N-1))^2 / 8).
    """
    c = check_coefficients(coefficients)
    grid_max = float(np.max(np.abs(np.fft.fft(c, fft_points)) ** 2))
    h = 2.0 * np.pi / fft_points
    slack = (h * (N - 1)) ** 2 / 8.0
    return float(math.sqrt(grid_max / (1.0 - slack)) / NORM) * (1.0 + 1e-12)


def canonical(coefficients) -> tuple[int, ...]:
    """Canonical representative under the score-preserving group of order 8.

    Generators: global negation, coefficient reversal, and c_k -> (-1)^k c_k (rotation by pi).
    """
    c = np.asarray(coefficients, dtype=np.int64)
    alt = np.where(np.arange(N) % 2 == 0, 1, -1)
    best = None
    for rev in (False, True):
        a = c[::-1] if rev else c
        for al in (False, True):
            b = a * alt if al else a
            for sg in (1, -1):
                t = tuple(int(v) for v in (sg * b))
                if best is None or t > best:
                    best = t
    return best
