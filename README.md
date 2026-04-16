# Enhanced Sharpe Ratio Inference with Sensitivity Decay and Strategy Rotation

## Overview

This repository extends the statistical framework established in **"Sharpe Ratio Inference under GARCH Returns"** (SSRN, 2026) by integrating a novel decision-theoretic layer: the **Jessicka Formulation** for sensitivity decay, edge rotation, and biological barriers in trading strategies.

While the original SSRN paper provides a rigorous method for calculating the asymptotic variance of the Sharpe ratio under heavy-tailed GARCH processes, it treats the strategy as static. This project addresses the critical gap: **how to act when the theoretical variance is infinite or unstable?** By modeling the trader's "edge" as a decaying biological response to repeated market exposure, we derive optimal rotation thresholds that maximize information gain and preserve capital in non-stationary regimes.

### Key Contributions
1.  **Unified Framework**: Blends stochastic volatility modeling (GARCH) with neural adaptation dynamics (Power-Law Decay).
2.  **Decision Rule**: Derives a mathematically grounded rotation trigger $\theta$ based on optimal information gain theory ($0.3 \leq \theta \leq 0.6$).
3.  **Reproducibility**: Provides a complete Jupyter notebook reproducing SSRN baseline results while demonstrating the out-of-sample superiority of the rotation-aware strategy.

---

## Theoretical Background

### 1. The Baseline: SSRN GARCH Inference

The foundational work, **"A Closed-Form Solution for Sharpe Ratio Inference under GARCH Returns"** by **López de Prado, Porcu, Zoonekynd, and Engle (2026)**, establishes that for returns $r_t$ following a GARCH(1,1) process with standardized innovations having tail index $\kappa$:

$$ r_t = \sigma_t z_t, \quad \sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2 $$

where $z_t$ follows a distribution with power-law tails $P(|z| > x) \sim c x^{-\kappa}$.

**Critical Insight**: When $2 < \kappa < 4$, the fourth moment of returns is infinite ($E[r_t^4] = \infty$). Consequently, the standard asymptotic variance of the Sharpe ratio estimator diverges. The SSRN paper derives a corrected asymptotic variance $V_{GARCH}$ (Formula 15) dependent on $\kappa$:

$$ \sqrt{T}(\hat{SR} - SR) \xrightarrow{d} N(0, V_{GARCH}) $$

However, this inference assumes a **static strategy**. It tells us *how uncertain* our estimate is, but not *when to stop* using the strategy as market dynamics evolve.

### 2. The Extension: The Jessicka Formulation

The **[Sensitivity Decay Whitepaper](https://github.com/algomaschine/sensitivity-decay-trading/blob/main/docs/WHITEPAPER_EN.md)** posits that trading edge is not a fixed parameter but a dynamic state variable subject to **habituation**. Just as biological systems desensitize to repeated stimuli, a strategy's alpha decays with repeated exposure to the same market regime.

#### A. Power-Law Edge Decay
Instead of assuming constant expected return $\mu$, we model the effective edge $\mu_{eff}(n)$ after $n$ exposures (trades) as:

$$ \sigma(n) = (1 + \beta n)^{-\eta} $$
$$ \mu_{eff}(n) = \bar{\mu} \cdot \sigma(n) $$

Where:
*   $\bar{\mu}$ is the **ceiling edge** (maximum initial alpha).
*   $\eta$ is the decay exponent, linked directly to the tail index from the SSRN model:
    $$ \eta = 1 - \frac{2}{\kappa} $$
    *Rationale*: Heavier tails (lower $\kappa$) imply faster structural instability and quicker edge decay.

#### B. Optimal Rotation Threshold
Drawing from **optimal information gain** theory (eLife, 2025), staying in a decaying strategy too long wastes energy (drawdown), while rotating too early incurs transaction costs and loses information. The optimal rotation threshold $\theta$ lies in the intermediate range:

$$ \text{Rotate if } \sigma(n) < \theta, \quad \theta \in [0.3, 0.6] $$

#### C. Overload Threshold & Position Sizing
To account for market-wide arousal (volatility clustering), we adjust the entry threshold $\tau_t$ and position size $w_t$:

$$ \tau_t = \tau_0 \left( 1 + \alpha_{load} \frac{\sigma_t}{\bar{\sigma}} \right) $$
$$ w_t \propto \sigma(n) $$

This ensures we trade smaller sizes as the edge decays and require stronger signals during high-volatility regimes.

---

## Methodology: Blending the Models

This project implements a two-stage simulation pipeline:

### Stage 1: SSRN Baseline Reproduction
We simulate $N$ paths of GARCH(1,1) returns with $\kappa \in (2, 4)$.
1.  Estimate parameters $(\omega, \alpha, \beta, \kappa)$ using `functions.py`.
2.  Compute the sample variance of the Sharpe ratio across paths.
3.  Verify convergence to the theoretical $V_{GARCH}$ from Formula 15.
4.  **Result**: Confirms that standard inference fails without tail correction.

### Stage 2: Jessicka Rotation Overlay
Using the *same* simulated paths:
1.  **Calibration**: Estimate $\bar{\mu}$ and $\kappa$ from the first 100 trades (training window).
2.  **Decay Calculation**: Compute $\eta = 1 - 2/\kappa$ and track $\sigma(n)$ in real-time.
3.  **Execution**:
    *   If $\sigma(n) < \theta$: **Rotate** (stop strategy, reset P&L).
    *   If $|r_t| < \tau_t$: **Skip** (overload filter).
    *   Else: Trade with size $w_t \propto \sigma(n)$.
4.  **Evaluation**: Compare the final Sharpe ratio and Maximum Drawdown of the **Rotating Strategy** vs. **Buy-and-Hold**.

---

## Repository Structure

```text
.
├── functions.py                  # Core GARCH utilities (SSRN)
├── Jessicka_Enhanced_Sharpe_Ratio.ipynb  # Main analysis notebook
├── Descriptive_Statistics.ipynb  # Original SSRN descriptive analysis
├── Simulation_with_skewness.ipynb # Original SSRN simulation baseline
├── README.md                     # This file
├── requirements.txt              # Dependencies
└── output/                       # Generated plots and summary stats
    ├── figure_1_ssrn_baseline.png
    ├── panel_b_sharpe_distribution.png
    ├── panel_c_decay_curve.png
    ├── panel_d_theta_sensitivity.png
    └── summary.txt
```

## Installation & Usage

### Prerequisites
Python 3.9+, with the following packages:
```bash
pip install numpy pandas matplotlib scipy arch statsmodels ray tqdm notebook papermill
```

### Running the Analysis
1.  Ensure `functions.py` is in the same directory.
2.  Launch Jupyter:
    ```bash
    jupyter notebook Jessicka_Enhanced_Sharpe_Ratio.ipynb
    ```
3.  Run all cells. The notebook will:
    *   Reproduce Figure 1 from the SSRN paper.
    *   Simulate the rotation strategy.
    *   Generate comparative panels (A-D).
    *   Save high-resolution PNGs to the `output/` folder.

### Alternative: Batch Execution with Papermill
```bash
mkdir -p outputs
papermill Jessicka_Enhanced_Sharpe_Ratio.ipynb outputs/Jessicka_Enhanced_Sharpe_Ratio.ipynb
```

---

## Key Findings (Preliminary)

In regimes where $\kappa < 4$ (infinite fourth moment):
1.  **Variance Reduction**: The rotating strategy exhibits significantly lower variance in realized Sharpe ratios compared to buy-and-hold.
2.  **Drawdown Control**: By exiting when $\sigma(n) < 0.5$, the strategy avoids the "long tail" of decay where noise dominates signal.
3.  **Robustness**: The power-law decay model fits empirical trade sequences better than exponential decay, aligning with neural adaptation literature.

## Pre-Analysis Plan (PAP) Compliance

This project adheres to the **Double-Blind** protocol outlined in the Whitepaper:
*   **Pre-registered Parameters**: $\eta$ derivation, $\theta$ grid $[0.3, 0.5, 0.6]$, $\alpha_{load} = 0.5$.
*   **Data Splitting**: Training (first 100 obs), Calibration (next 200), Holdout (remainder).
*   **No Look-Ahead**: Decay states $\sigma(n)$ are computed strictly using past information.

## Limitations & Future Work

*   **Estimation Error**: The tail index $\kappa$ is noisy in finite samples; robust estimators (Hill estimator) are used but introduce variance.
*   **Regime Detection**: Current implementation uses a simple exposure count $n$. Future versions will integrate **Statistical Jump Models (JM)** to detect regime changes $C_t$ explicitly, resetting $n$ only when the regime persists.
*   **Real Data Validation**: While simulations match theoretical properties, validation on live hedge fund indices (e.g., HFR) is required to confirm the $\eta = 1 - 2/\kappa$ relationship.

## References

1.  **López de Prado, M., Porcu, E., Zoonekynd, V., & Engle, R. F.** (2026). *A Closed-Form Solution for Sharpe Ratio Inference under GARCH Returns*. SSRN. [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6568702)
2.  **Samokhvalov, E.** (2025). *From Arousal Decay to Edge Decay — A Unified Formulation for Markets*. GitHub. [Link](https://github.com/algomaschine/sensitivity-decay-trading/blob/main/docs/WHITEPAPER_EN.md)
3.  **Nature Communications** (2023). Power-law adaptation in primary visual cortex.
4.  **eLife** (2025). Optimal information gain at intermediate habituation.

---

**License**: MIT  
**Author**: Eduard Samokhvalov  
**Date**: 2025-02-19
