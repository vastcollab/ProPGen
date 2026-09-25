"""Valley-crossing relaxation time from bridge trajectories.

The approach to equilibrium of the far-side genotype-phenotype pair (2, 0) is
fitted to a decaying exponential by orthogonal distance regression,

.. math::
    -(f_2^{(0)}(t) - f^{\\mathrm{eq}}_{2,0}) = \\beta_0 e^{\\beta_1 t / S}

and the relaxation time is :math:`\\tau = -S / \\beta_1`. This is the quantity
plotted against the bridge probability :math:`\\phi`: a lower :math:`\\tau`
means the population crosses the fitness valley sooner.
"""

from __future__ import annotations

import numpy as np
from scipy import odr

__all__ = ["FAR_SIDE_PAIR", "TIME_SCALE", "fit_tau", "tau_vs_phi"]

#: The genotype-phenotype pair on the far side of the valley.
FAR_SIDE_PAIR = (2, 0)

#: The fit is performed on t / TIME_SCALE to keep the exponent near unity,
#: which conditions the regression. tau is converted back to cycles.
TIME_SCALE = 10_000.0


def _exponential(beta, x):
    return beta[0] * np.exp(beta[1] * x)


def fit_tau(trajectory: np.ndarray, f_eq_value: float, cycles: np.ndarray) -> dict:
    """Fit the approach to equilibrium of one trajectory.

    Parameters
    ----------
    trajectory
        Mean frequency of the far-side pair at each recorded cycle.
    f_eq_value
        Analytic equilibrium frequency of that pair.
    cycles
        Dilution-cycle index of each recorded point.

    Returns
    -------
    dict with ``tau`` (relaxation time in dilution cycles), ``beta0``,
    ``beta1`` and ``residual_variance``.
    """
    deficit = -(np.asarray(trajectory, dtype=np.float64) - f_eq_value)
    scaled = np.asarray(cycles, dtype=np.float64) / TIME_SCALE

    fit = odr.ODR(
        odr.Data(scaled, deficit),
        odr.Model(_exponential),
        beta0=[1.0, -0.001],
    ).run()

    beta0, beta1 = fit.beta
    return {
        "tau": float(-TIME_SCALE / beta1) if beta1 != 0 else float("inf"),
        "beta0": float(beta0),
        "beta1": float(beta1),
        "residual_variance": float(fit.res_var),
    }


def tau_vs_phi(summary, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Relaxation time as a function of bridge probability, at one valley depth.

    ``summary`` is the aggregated bridge summary. Its stored coordinate is
    ``bridge_mapping``; the bridge probability plotted in the paper is
    ``phi = 1 - bridge_mapping``.
    """
    g, p = FAR_SIDE_PAIR
    n_phenotypes = summary.shape[1]
    flat_index = g * n_phenotypes + p

    phis, taus = [], []
    for mapping in summary.values("bridge_mapping"):
        panel = summary.select(bridge_mapping=mapping, gamma=gamma)
        fit = fit_tau(panel["mean"][g, p], panel["f_eq"][flat_index], panel["cycles"])
        phis.append(1.0 - mapping)
        taus.append(fit["tau"])

    order = np.argsort(phis)
    return np.asarray(phis)[order], np.asarray(taus)[order]
