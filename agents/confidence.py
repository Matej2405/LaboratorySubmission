"""Statistical confidence helpers.

* `wilson_interval(k, n, alpha)` - Wilson score interval for a binomial
  proportion. Robust at small n and tail proportions.
* `beta_posterior(k, n, alpha, beta)` - Beta(alpha+k, beta+n-k) posterior mean
  and 5/95% credible interval. Used per-facility, where k=#agreeing validator
  passes, n=total passes, prior alpha=beta=1 (uniform).

No SciPy dependency at import time so we can use these inside Spark UDFs;
SciPy is imported lazily where its statistical tables are needed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
    n: int

    def fmt(self, pct: bool = True) -> str:
        if pct:
            return f"{self.point*100:.1f}% [{self.lower*100:.1f}-{self.upper*100:.1f}%] (n={self.n})"
        return f"{self.point:.3f} [{self.lower:.3f}-{self.upper:.3f}] (n={self.n})"


def _z(alpha: float) -> float:
    """Two-sided z critical value for confidence 1-alpha."""
    table = {0.10: 1.6449, 0.05: 1.96, 0.01: 2.5758}
    return table.get(round(alpha, 2), 1.96)


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return Interval(point=0.0, lower=0.0, upper=1.0, n=0)
    z = _z(alpha)
    p_hat = k / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denom
    return Interval(
        point=p_hat,
        lower=max(0.0, centre - margin),
        upper=min(1.0, centre + margin),
        n=n,
    )


def beta_posterior(
    k: int, n: int, *, prior_alpha: float = 1.0, prior_beta: float = 1.0, alpha: float = 0.10
) -> Interval:
    """Beta credible interval (default 90% CI). Uses scipy if available."""
    a = prior_alpha + k
    b = prior_beta + max(0, n - k)
    mean = a / (a + b)
    try:
        from scipy.stats import beta as scipy_beta  # type: ignore
        lo = float(scipy_beta.ppf(alpha / 2, a, b))
        hi = float(scipy_beta.ppf(1 - alpha / 2, a, b))
    except Exception:
        # Coarse normal approximation fallback
        var = (a * b) / (((a + b) ** 2) * (a + b + 1))
        sd = math.sqrt(var)
        z = _z(alpha)
        lo, hi = max(0.0, mean - z * sd), min(1.0, mean + z * sd)
    return Interval(point=mean, lower=lo, upper=hi, n=n)


def trust_weighted_proportion(
    weights: list[float], indicators: list[int], alpha: float = 0.05
) -> Interval:
    """Confidence interval for a trust-weighted proportion.

    `weights[i]` is the trust score in [0,1]; `indicators[i]` is 0/1 for the
    capability being claimed-and-functional.

    Uses the effective-sample-size approximation: n_eff = (sum w)^2 / sum(w^2).
    """
    if not weights or not indicators or len(weights) != len(indicators):
        return Interval(point=0.0, lower=0.0, upper=1.0, n=0)
    sw = sum(weights) or 1e-9
    sw2 = sum(w * w for w in weights) or 1e-9
    n_eff = max(1, int(sw * sw / sw2))
    p_hat = sum(w * x for w, x in zip(weights, indicators)) / sw
    return wilson_interval(int(round(p_hat * n_eff)), n_eff, alpha=alpha)


@dataclass(frozen=True)
class DesertIndex:
    """Population per facility offering a capability, with Wilson bounds.

    `point` is the headline ratio (people per 100 k served per capable
    facility). `lower` and `upper` come from the Wilson interval on the
    capability prevalence p-hat and bracket how many capable facilities the
    population could realistically expect.
    """
    point: float
    lower: float
    upper: float
    population: int
    n_facilities: int
    n_capable_eff: float
    p_hat: float

    def fmt(self) -> str:
        return (
            f"{self.point:,.0f} ppl/100k per capable facility "
            f"[{self.lower:,.0f}-{self.upper:,.0f}] "
            f"(pop={self.population:,}, capable~{self.n_capable_eff:.1f}/{self.n_facilities})"
        )


def desert_index(
    population: int,
    n_facilities: int,
    p_hat: float,
    *,
    alpha: float = 0.05,
    floor: float = 0.1,
) -> DesertIndex:
    """Population per 100 k served per capable facility, with Wilson bounds.

    A larger value = bigger crisis. Uses Wilson on the binomial (k, n) where
    `k = round(p_hat * n)` and `n = n_facilities` to bracket how many capable
    facilities the population can rely on.
    """
    n = max(1, int(n_facilities))
    k = max(0, min(n, int(round(p_hat * n))))
    iv = wilson_interval(k, n, alpha=alpha)
    pop_per_100k = max(1.0, population / 100_000.0)
    point = pop_per_100k / max(floor, p_hat * n)
    lower = pop_per_100k / max(floor, iv.upper * n)
    upper = pop_per_100k / max(floor, iv.lower * n)
    return DesertIndex(
        point=point,
        lower=lower,
        upper=upper,
        population=int(population),
        n_facilities=n,
        n_capable_eff=p_hat * n,
        p_hat=p_hat,
    )
