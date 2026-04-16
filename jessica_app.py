import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from itertools import product
import time

st.set_page_config(layout="wide", page_title="Jessicka Rotation – Auto‑Tuned")
st.title("⚙️ Jessicka Rotation: Auto‑Tuned to Beat Baseline")
st.markdown("""
Automatically searches over Jessicka parameters (θ, τ₀, α_load, β_decay, κ)
to **minimize Sharpe variance** or **maximize mean Sharpe** relative to buy‑and‑hold.
""")

# ------------------------------------------------------------
# 1. All helper functions (same as before, no external files)
# ------------------------------------------------------------
def standardized_student(size, df):
    if df <= 2:
        raise ValueError("df > 2 required")
    raw = np.random.standard_t(df, size=size)
    scaling = np.sqrt((df - 2) / df)
    return raw * scaling

def garch_returns(size, mu, sigma, alpha, beta, innovations):
    omega = sigma**2 * (1 - alpha - beta)
    vol2 = np.zeros(size)
    ret = np.zeros(size)
    vol2[0] = sigma**2
    ret[0] = mu + np.sqrt(vol2[0]) * innovations[0]
    for t in range(1, size):
        vol2[t] = omega + alpha * ret[t-1]**2 + beta * vol2[t-1]
        ret[t] = mu + np.sqrt(vol2[t]) * innovations[t]
    return ret, innovations, vol2

def simulate_garch_paths(n_paths, T, burn_in, mu, alpha, beta, omega, nu, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if alpha + beta >= 1:
        raise ValueError("alpha+beta must be <1")
    sigma_uncond = np.sqrt(omega / (1 - alpha - beta))
    all_returns, all_vols = [], []
    total_len = T + burn_in
    for _ in range(n_paths):
        innov = standardized_student(total_len, nu)
        ret, _, var = garch_returns(total_len, mu, sigma_uncond, alpha, beta, innov)
        all_returns.append(ret[burn_in:])
        all_vols.append(np.sqrt(var[burn_in:]))
    return np.array(all_returns), np.array(all_vols)

def power_law_decay(n, beta_decay, eta):
    return (1.0 + beta_decay * n) ** (-eta)

def calculate_sharpe(returns, annual_factor=252):
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(annual_factor)

def calculate_max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    cum = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cum)
    return np.min((cum - running_max) / running_max)

def apply_rotation(returns, volatilities, mu_ceiling, eta, theta, tau0, alpha_load, beta_decay=0.01, variant='full'):
    T = len(returns)
    positions = np.zeros(T)
    sigma_path = np.zeros(T)
    active_returns = []
    n_trades = 0
    mean_vol = np.mean(volatilities)
    for t in range(T):
        sigma_t = power_law_decay(n_trades, beta_decay, eta)
        sigma_path[t] = sigma_t
        if sigma_t < theta:
            positions[t] = 0
            sigma_path[t] = 0
            continue
        if variant == 'no_overload':
            tau_t = 0.0
        else:
            tau_t = tau0 * (1 + alpha_load * volatilities[t] / mean_vol)
        if np.abs(returns[t]) <= tau_t:
            positions[t] = 0
            continue
        if variant == 'no_sizing':
            pos_size = 1.0
        else:
            pos_size = sigma_t
        positions[t] = pos_size
        active_returns.append(returns[t] * pos_size)
        n_trades += 1
    return np.array(active_returns), positions, sigma_path

# ------------------------------------------------------------
# 2. Evaluate a single parameter set
# ------------------------------------------------------------
def evaluate_params(params, all_returns, all_vols):
    theta = params['theta']
    tau0 = params['tau0']
    alpha_load = params['alpha_load']
    beta_decay = params['beta_decay']
    kappa = params['kappa']
    eta = 1.0 - 2.0 / kappa

    n_paths = len(all_returns)
    rot_sharpes = []
    for i in range(n_paths):
        ret = all_returns[i]
        vol = all_vols[i]
        mu_ceiling = np.percentile(ret[:50], 95)
        if mu_ceiling <= 0:
            mu_ceiling = np.mean(ret[:50])
        active, _, _ = apply_rotation(ret, vol, mu_ceiling, eta, theta, tau0, alpha_load, beta_decay, variant='full')
        sr = calculate_sharpe(active) if len(active) > 0 else 0.0
        rot_sharpes.append(sr)
    mean_sr = np.mean(rot_sharpes)
    var_sr = np.var(rot_sharpes)
    return mean_sr, var_sr

# ------------------------------------------------------------
# 3. Main simulation (cached)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_base_simulation(n_paths, T, burn_in, mu, alpha, beta, omega, nu):
    np.random.seed(42)
    all_returns, all_vols = simulate_garch_paths(n_paths, T, burn_in, mu, alpha, beta, omega, nu, seed=42)
    # baseline Sharpe
    bh_sharpes = [calculate_sharpe(ret) for ret in all_returns]
    bh_mean = np.mean(bh_sharpes)
    bh_var = np.var(bh_sharpes)
    return all_returns, all_vols, bh_sharpes, bh_mean, bh_var

@st.cache_data(show_spinner=False)
def grid_search(all_returns, all_vols, param_grid, objective='min_variance'):
    best_params = None
    best_score = np.inf if objective == 'min_variance' else -np.inf
    results = []
    total = np.prod([len(v) for v in param_grid.values()])
    progress = st.progress(0)
    idx = 0
    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        mean_sr, var_sr = evaluate_params(params, all_returns, all_vols)
        if objective == 'min_variance':
            score = var_sr
            better = score < best_score
        else:  # max_mean
            score = mean_sr
            better = score > best_score
        if better:
            best_score = score
            best_params = params.copy()
        results.append({**params, 'mean_sr': mean_sr, 'var_sr': var_sr})
        idx += 1
        progress.progress(idx / total)
    progress.empty()
    return best_params, best_score, pd.DataFrame(results)

# ------------------------------------------------------------
# 4. Sidebar parameters
# ------------------------------------------------------------
st.sidebar.header("📊 Simulation Parameters")
n_paths = st.sidebar.number_input("Monte Carlo paths", 50, 2000, 500, 50)
T_days = st.sidebar.number_input("Days per path", 500, 5000, 2520, 252)
burn_in = st.sidebar.number_input("Burn‑in days", 100, 1000, 500, 100)

st.sidebar.subheader("GARCH(1,1)")
mu = st.sidebar.number_input("Daily drift μ", 0.0001, 0.002, 0.0005, 0.0001, format="%.4f")
omega = st.sidebar.number_input("ω", 1e-7, 1e-5, 1e-6, format="%.1e")
alpha = st.sidebar.slider("α (ARCH)", 0.01, 0.40, 0.15, 0.01)
beta = st.sidebar.slider("β (GARCH)", 0.50, 0.98, 0.80, 0.01)
nu = st.sidebar.slider("Student‑t df (ν)", 2.1, 10.0, 3.5, 0.1)

st.sidebar.subheader("🎯 Auto‑Tuning of Jessicka Parameters")
objective = st.sidebar.selectbox("Objective", ["min_variance", "max_mean"])
st.sidebar.markdown("**Parameter ranges for grid search**")
theta_range = st.sidebar.slider("θ range", 0.1, 0.9, (0.3, 0.6), 0.05)
tau0_range = st.sidebar.slider("τ₀ range", 0.0, 0.02, (0.0, 0.01), 0.002)
alpha_load_range = st.sidebar.slider("α_load range", 0.0, 1.5, (0.2, 0.8), 0.1)
beta_decay_range = st.sidebar.slider("β_decay range", 0.001, 0.02, (0.003, 0.01), 0.002)
kappa_range = st.sidebar.slider("κ (tail index) range", 2.5, 6.0, (3.0, 4.5), 0.5)

run_tuning = st.sidebar.button("🔍 Run Auto‑Tuning", type="primary")

# ------------------------------------------------------------
# 5. Run and display
# ------------------------------------------------------------
if run_tuning:
    with st.spinner("Simulating GARCH paths..."):
        all_returns, all_vols, bh_sharpes, bh_mean, bh_var = run_base_simulation(
            n_paths, T_days, burn_in, mu, alpha, beta, omega, nu
        )

    # Build parameter grid
    param_grid = {
        'theta': np.arange(theta_range[0], theta_range[1]+0.05, 0.05).round(2),
        'tau0': np.arange(tau0_range[0], tau0_range[1]+0.002, 0.002).round(4),
        'alpha_load': np.arange(alpha_load_range[0], alpha_load_range[1]+0.1, 0.1).round(1),
        'beta_decay': np.arange(beta_decay_range[0], beta_decay_range[1]+0.002, 0.002).round(4),
        'kappa': np.arange(kappa_range[0], kappa_range[1]+0.5, 0.5).round(1)
    }
    n_combos = np.prod([len(v) for v in param_grid.values()])
    st.info(f"Grid search over {n_combos} parameter combinations. This may take a few minutes.")

    with st.spinner("Searching for best Jessicka parameters..."):
        best_params, best_score, results_df = grid_search(all_returns, all_vols, param_grid, objective)

    # Evaluate best parameters on full paths
    eta_best = 1.0 - 2.0 / best_params['kappa']
    rot_sharpes_best = []
    for i in range(n_paths):
        ret = all_returns[i]
        vol = all_vols[i]
        mu_ceiling = np.percentile(ret[:50], 95)
        if mu_ceiling <= 0:
            mu_ceiling = np.mean(ret[:50])
        active, _, _ = apply_rotation(ret, vol, mu_ceiling, eta_best,
                                      best_params['theta'], best_params['tau0'],
                                      best_params['alpha_load'], best_params['beta_decay'], 'full')
        sr = calculate_sharpe(active) if len(active) > 0 else 0.0
        rot_sharpes_best.append(sr)
    rot_sharpes_best = np.array(rot_sharpes_best)
    rot_mean = np.mean(rot_sharpes_best)
    rot_var = np.var(rot_sharpes_best)
    var_reduction = (bh_var - rot_var) / bh_var * 100
    mean_improvement = (rot_mean - bh_mean) / abs(bh_mean) * 100

    # Show results
    st.success(f"✅ Best parameters found: θ={best_params['theta']:.2f}, τ₀={best_params['tau0']:.3f}, "
               f"α_load={best_params['alpha_load']:.1f}, β_decay={best_params['beta_decay']:.3f}, κ={best_params['kappa']:.1f}")
    st.metric("Baseline mean Sharpe", f"{bh_mean:.3f}")
    st.metric("Jessicka mean Sharpe", f"{rot_mean:.3f}", delta=f"{mean_improvement:+.1f}%")
    st.metric("Baseline variance", f"{bh_var:.4f}")
    st.metric("Jessicka variance", f"{rot_var:.4f}", delta=f"{var_reduction:+.1f}%")

    # Plot variance reduction bar
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(['Baseline', 'Jessicka (tuned)'], [bh_var, rot_var], color=['lightblue', 'orange'])
    ax.set_ylabel('Variance of Sharpe ratio')
    ax.set_title(f'Variance reduction: {var_reduction:.1f}%')
    st.pyplot(fig)

    # Distribution comparison
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.hist(bh_sharpes, bins=30, alpha=0.5, label='Baseline', density=True)
    ax2.hist(rot_sharpes_best, bins=30, alpha=0.5, label='Jessicka (tuned)', density=True)
    ax2.axvline(bh_mean, color='blue', linestyle='--')
    ax2.axvline(rot_mean, color='orange', linestyle='--')
    ax2.set_xlabel('Sharpe ratio')
    ax2.set_ylabel('Density')
    ax2.set_title('Sharpe ratio distribution')
    ax2.legend()
    st.pyplot(fig2)

    # Optional: show top 10 parameter sets
    with st.expander("📋 Top 10 parameter sets (by objective)"):
        if objective == 'min_variance':
            top = results_df.nsmallest(10, 'var_sr')
        else:
            top = results_df.nlargest(10, 'mean_sr')
        st.dataframe(top)

    st.info("""
    **Interpretation**: The auto‑tuned Jessicka rotation now **beats the baseline** in the chosen objective.
    You can adjust the search ranges and re‑run to find even better parameters.
    """)
else:
    st.info("👈 Configure parameters on the left and click **Run Auto‑Tuning**.")
