"""Closed-form persister resuscitation and partitioning dynamics.

The diffusion limit of ProP Gen theory applied to bacterial persistence.
A population of dormant persisters (P) resuscitates at an accelerating rate
:math:`S(t) = \\alpha e^{\\beta t}`. On resuscitating, a persister partitions
into one of three fates with fixed probabilities: failed (F), damaged (D) or
healthy (H). Damaged cells grow at :math:`X_D` and convert to healthy at
division with probability :math:`\\phi_{DH}`; healthy cells grow at
:math:`X_H`.

Solving the resulting linear system gives

.. math::
    n_P(t) &= n_P(0)\\, e^{-c(e^{\\beta t} - 1)} \\\\
    n_F(t) &= n_F(0) + \\frac{\\sigma_{PF}}{\\Sigma_\\sigma}(n_P(0) - n_P(t))

with :math:`n_D` and :math:`n_H` expressed through lower incomplete gamma
functions, where :math:`c = \\alpha \\Sigma_\\sigma / \\beta`.

This figure requires no simulation and no stored data.

The partitioning probabilities are taken from Fang & Allison's measurements of
persister resuscitation.
"""

from __future__ import annotations

import numpy as np
from mpmath import gammainc

__all__ = ["PARAMETERS", "compartments"]

#: Default parameters. The sigma_P* values are from Fang & Allison; the growth
#: and resuscitation rates are the estimates used in the paper.
PARAMETERS = {
    "alpha": 0.5,       # base resuscitation rate
    "beta": 1.0,        # acceleration of resuscitation, S(t) = alpha e^{beta t}
    "X_H": 0.8,         # growth rate of healthy cells
    "X_D": 0.4,         # growth rate of damaged cells
    "phi_DH": 0.3,      # probability damaged -> healthy at division
    "sigma_PF": 0.64,   # persister -> failed
    "sigma_PD": 0.32,   # persister -> damaged
    "sigma_PH": 0.04,   # persister -> healthy
}


def _lower_incomplete_gamma(s: float, x: float) -> float:
    return float(gammainc(s, 0, x))


def compartments(
    t: np.ndarray,
    *,
    alpha: float = PARAMETERS["alpha"],
    beta: float = PARAMETERS["beta"],
    X_H: float = PARAMETERS["X_H"],
    X_D: float = PARAMETERS["X_D"],
    phi_DH: float = PARAMETERS["phi_DH"],
    sigma_PF: float = PARAMETERS["sigma_PF"],
    sigma_PD: float = PARAMETERS["sigma_PD"],
    sigma_PH: float = PARAMETERS["sigma_PH"],
    n0: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """Population fractions of each compartment over time.

    Parameters
    ----------
    t
        Times at which to evaluate the solution.
    n0
        Initial populations ``(P, F, D, H)``. The default starts as all
        persisters, matching a culture emerging from antibiotic treatment.
    normalize
        Return fractions of the total population rather than absolute numbers.

    Returns
    -------
    dict with keys ``P``, ``F``, ``D``, ``H``.
    """
    t = np.asarray(t, dtype=float)
    nP0, nF0, nD0, nH0 = n0

    Sigma_sigma = sigma_PF + sigma_PD + sigma_PH
    a = X_D * (1.0 - phi_DH)
    c = alpha * Sigma_sigma / beta
    q_D = 1.0 - a / beta
    q_H = 1.0 - X_H / beta
    k = X_D * phi_DH / (X_H - a)
    B = sigma_PH + k * sigma_PD

    nP = np.empty_like(t)
    nF = np.empty_like(t)
    nD = np.empty_like(t)
    nH = np.empty_like(t)

    for i, ti in enumerate(t):
        nP_t = nP0 * np.exp(-c * (np.exp(beta * ti) - 1.0))
        nF_t = nF0 + (sigma_PF / Sigma_sigma) * (nP0 - nP_t)

        ce_bt = c * np.exp(beta * ti)
        term_D = _lower_incomplete_gamma(q_D, ce_bt) - _lower_incomplete_gamma(q_D, c)
        term_H = _lower_incomplete_gamma(q_H, ce_bt) - _lower_incomplete_gamma(q_H, c)

        damaged_bracket = (
            nD0 + (alpha * sigma_PD * nP0 / beta) * np.exp(c) * (c ** (-q_D)) * term_D
        )
        nD_t = np.exp(a * ti) * damaged_bracket
        nH_t = np.exp(X_H * ti) * (
            nH0 + k * nD0 + (alpha * B * nP0 / beta) * np.exp(c) * (c ** (-q_H)) * term_H
        ) - k * np.exp(a * ti) * damaged_bracket

        nP[i], nF[i], nD[i], nH[i] = nP_t, nF_t, nD_t, nH_t

    if normalize:
        total = nP + nF + nD + nH
        nP, nF, nD, nH = nP / total, nF / total, nD / total, nH / total

    return {"P": nP, "F": nF, "D": nD, "H": nH}
