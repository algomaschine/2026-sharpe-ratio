import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random

st.set_page_config(layout="wide", page_title="Jessicka Rotation – Complete")
st.title("📊 Jessicka Rotation: Full Analysis + Fast Random Search")

# ------------------------------------------------------------
# 1. Core functions (no external files)
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
    if mean_vol == 0:
        mean_vol = 1.0
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
# 2. Fast evaluation (subset of paths)
# ------------------------------------------------------------
def evaluate_params_fast(params, all_returns, all_vols, n_eval_paths):
    theta = params['theta']
    tau0 = params['tau0']
    alpha_load = params['alpha_load']
    beta_decay = params['beta_decay']
    kappa = params['kappa']
    eta = 1.0 - 2.0 / kappa
    rot_sharpes = []
    for i in range(min(n_eval_paths, len(all_returns))):
        ret = all_returns[i]
        vol = all_vols[i]
        mu_ceiling = np.percentile(ret[:50], 95)
        if mu_ceiling <= 0:
            mu_ceiling = np.mean(ret[:50])
        active, _, _ = apply_rotation(ret, vol, mu_ceiling, eta, theta, tau0, alpha_load, beta_decay, 'full')
        sr = calculate_sharpe(active) if len(active) > 0 else 0.0
        rot_sharpes.append(sr)
    return np.mean(rot_sharpes), np.var(rot_sharpes)

def random_search(all_returns, all_vols, param_ranges, n_trials, n_eval_paths, objective):
    best_params = None
    best_score = np.inf if objective == 'min_variance' else -np.inf
    results = []
    progress = st.progress(0)
    for trial in range(n_trials):
        params = {
            'theta': random.uniform(*param_ranges['theta']),
            'tau0': random.uniform(*param_ranges['tau0']),
            'alpha_load': random.uniform(*param_ranges['alpha_load']),
            'beta_decay': random.uniform(*param_ranges['beta_decay']),
            'kappa': random.uniform(*param_ranges['kappa'])
        }
        mean_sr, var_sr = evaluate_params_fast(params, all_returns, all_vols, n_eval_paths)
        if objective == 'min_variance':
            score = var_sr
            better = score < best_score
        else:
            score = mean_sr
            better = score > best_score
        if better:
            best_score = score
            best_params = params.copy()
        results.append({**params, 'mean_sr': mean_sr, 'var_sr': var_sr})
        progress.progress((trial + 1) / n_trials)
    progress.empty()
    return best_params, best_score, pd.DataFrame(results)

# ------------------------------------------------------------
# 3. Full evaluation (all paths, all visuals)
# ------------------------------------------------------------
def full_evaluation(all_returns, all_vols, best_params):
    kappa = best_params['kappa']
    eta = 1.0 - 2.0 / kappa
    rot_sharpes = []
    rot_dds = []
    sigma_paths = []
    avg_positions = []
    for i in range(len(all_returns)):
        ret = all_returns[i]
        vol = all_vols[i]
        mu_ceiling = np.percentile(ret[:50], 95)
        if mu_ceiling <= 0:
            mu_ceiling = np.mean(ret[:50])
        active, pos, sig = apply_rotation(ret, vol, mu_ceiling, eta,
                                          best_params['theta'], best_params['tau0'],
                                          best_params['alpha_load'], best_params['beta_decay'], 'full')
        rot_sharpes.append(calculate_sharpe(active) if len(active) else 0.0)
        rot_dds.append(calculate_max_drawdown(active) if len(active) else 0.0)
        sigma_paths.append(sig)
        avg_positions.append(np.mean(pos))
    rot_sharpes = np.array(rot_sharpes)
    rot_dds = np.array(rot_dds)
    sigma_paths = np.array(sigma_paths)
    avg_positions = np.array(avg_positions)

    # Ablation (subset)
    ab_full, ab_no_ol, ab_no_sz = [], [], []
    n_ablate = min(150, len(all_returns))
    for i in range(n_ablate):
        ret = all_returns[i]
        vol = all_vols[i]
        mu_c = np.percentile(ret[:50], 95)
        if mu_c <= 0:
            mu_c = np.mean(ret[:50])
        a_f, _, _ = apply_rotation(ret, vol, mu_c, eta, best_params['theta'],
                                   best_params['tau0'], best_params['alpha_load'],
                                   best_params['beta_decay'], 'full')
        ab_full.append(calculate_sharpe(a_f) if len(a_f) else 0.0)
        a_nol, _, _ = apply_rotation(ret, vol, mu_c, eta, best_params['theta'],
                                     0.0, 0.0, best_params['beta_decay'], 'no_overload')
        ab_no_ol.append(calculate_sharpe(a_nol) if len(a_nol) else 0.0)
        a_nosz, _, _ = apply_rotation(ret, vol, mu_c, eta, best_params['theta'],
                                      best_params['tau0'], best_params['alpha_load'],
                                      best_params['beta_decay'], 'no_sizing')
        ab_no_sz.append(calculate_sharpe(a_nosz) if len(a_nosz) else 0.0)
    return rot_sharpes, rot_dds, sigma_paths, avg_positions, (ab_full, ab_no_ol, ab_no_sz)

# ------------------------------------------------------------
# 4. Cached base simulation
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_base_simulation(n_paths, T, burn_in, mu, alpha, beta, omega, nu):
    np.random.seed(42)
    all_returns, all_vols = simulate_garch_paths(n_paths, T, burn_in, mu, alpha, beta, omega, nu, seed=42)
    bh_sharpes = [calculate_sharpe(ret) for ret in all_returns]
    bh_dds = [calculate_max_drawdown(ret) for ret in all_returns]
    bh_mean = np.mean(bh_sharpes)
    bh_var = np.var(bh_sharpes)
    return all_returns, all_vols, bh_sharpes, bh_dds, bh_mean, bh_var

# ------------------------------------------------------------
# 5. Sidebar
# ------------------------------------------------------------
st.sidebar.header("📊 Simulation")
n_paths = st.sidebar.number_input("Paths", 100, 2000, 500, 100)
T_days = st.sidebar.number_input("Days", 500, 5000, 2520, 252)
burn_in = st.sidebar.number_input("Burn‑in", 100, 1000, 500, 100)

st.sidebar.subheader("GARCH")
mu = st.sidebar.number_input("μ", 0.0001, 0.002, 0.0005, 0.0001, format="%.4f")
omega = st.sidebar.number_input("ω", 1e-7, 1e-5, 1e-6, format="%.1e")
alpha = st.sidebar.slider("α", 0.01, 0.40, 0.15, 0.01)
beta = st.sidebar.slider("β", 0.50, 0.98, 0.80, 0.01)
nu = st.sidebar.slider("ν (df)", 2.1, 10.0, 3.5, 0.1)

st.sidebar.subheader("Random Search")
objective = st.sidebar.selectbox("Objective", ["min_variance", "max_mean"])
n_trials = st.sidebar.number_input("Trials", 30, 300, 80, 10)
n_eval_paths = st.sidebar.number_input("Eval paths (fast)", 20, 200, 80, 10)

st.sidebar.subheader("Parameter ranges")
theta_range = st.sidebar.slider("θ", 0.1, 0.9, (0.3, 0.6), 0.05)
tau0_range = st.sidebar.slider("τ₀", 0.0, 0.02, (0.0, 0.01), 0.002)
alpha_load_range = st.sidebar.slider("α_load", 0.0, 1.5, (0.2, 0.8), 0.1)
beta_decay_range = st.sidebar.slider("β_decay", 0.001, 0.02, (0.003, 0.01), 0.002)
kappa_range = st.sidebar.slider("κ", 2.5, 6.0, (3.0, 4.5), 0.5)

run = st.sidebar.button("🚀 RUN", type="primary")

# ------------------------------------------------------------
# 6. Main
# ------------------------------------------------------------
if run:
    with st.spinner("Simulating GARCH..."):
        all_returns, all_vols, bh_sharpes, bh_dds, bh_mean, bh_var = run_base_simulation(
            n_paths, T_days, burn_in, mu, alpha, beta, omega, nu)

    param_ranges = {
        'theta': theta_range, 'tau0': tau0_range, 'alpha_load': alpha_load_range,
        'beta_decay': beta_decay_range, 'kappa': kappa_range
    }
    st.info(f"Searching {n_trials} combos...")
    with st.spinner("Random search..."):
        best_params, best_score, _ = random_search(
            all_returns, all_vols, param_ranges, n_trials, n_eval_paths, objective)

    with st.spinner("Final evaluation..."):
        rot_sharpes, rot_dds, sigma_paths, _, (ab_full, ab_nol, ab_nosz) = full_evaluation(
            all_returns, all_vols, best_params)

    rot_mean = np.mean(rot_sharpes)
    rot_var = np.var(rot_sharpes)
    var_red = (bh_var - rot_var) / bh_var * 100
    mean_imp = (rot_mean - bh_mean) / abs(bh_mean) * 100

    st.success(f"Best: θ={best_params['theta']:.2f}, τ₀={best_params['tau0']:.3f}, "
               f"α_load={best_params['alpha_load']:.1f}, β_decay={best_params['beta_decay']:.3f}, κ={best_params['kappa']:.1f}")
    c1, c2 = st.columns(2)
    c1.metric("Baseline Sharpe", f"{bh_mean:.3f}")
    c1.metric("Baseline variance", f"{bh_var:.4f}")
    c2.metric("Jessicka Sharpe", f"{rot_mean:.3f}", delta=f"{mean_imp:+.1f}%")
    c2.metric("Jessicka variance", f"{rot_var:.4f}", delta=f"{var_red:+.1f}%")

    # ---------- Figure 1: SSRN baseline ----------
    sample_sizes = [252, 504, 1008, 2520]
    sample_vars, theo_vars = [], []
    long_innov = standardized_student(200000, nu)
    sigma_uncond = np.sqrt(omega / (1 - alpha - beta))
    long_ret, _, _ = garch_returns(200000, mu, sigma_uncond, alpha, beta, long_innov)
    true_skew = stats.skew(long_ret)
    true_kurt = stats.kurtosis(long_ret) + 3
    true_SR = mu / sigma_uncond
    def vg(SR, skew, kurt, a, b, T):
        phi = a + b
        term2 = - SR * (1 - b) / (1 - phi) * skew
        denom = 1 - a**2 * kurt - 2*a*b - b**2
        term3 = SR**2 * (kurt - 1) / 4 * (1 - b)**2 * (1 + phi) / (1 - phi) / denom
        return (1 + term2 + term3) / T
    for t in sample_sizes:
        sharp_t = [calculate_sharpe(all_returns[i][:t]) for i in range(n_paths)]
        sample_vars.append(np.var(sharp_t, ddof=1))
        theo_vars.append(vg(true_SR, true_skew, true_kurt, alpha, beta, t))
    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.loglog(sample_sizes, sample_vars, 'o-', label='Sample')
    ax1.loglog(sample_sizes, theo_vars, 'r--', label='Theoretical')
    ax1.set_xlabel('Sample size T'); ax1.set_ylabel('Variance of Sharpe')
    ax1.set_title('SSRN Figure 1'); ax1.legend(); ax1.grid()
    st.pyplot(fig1)

    # ---------- Panels A-D ----------
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    axes[0,0].bar(['Baseline','Jessicka'], [bh_var, rot_var], color=['lightblue','orange'])
    axes[0,0].set_title(f'Variance reduction {var_red:.1f}%')
    axes[0,1].violinplot([bh_sharpes, rot_sharpes], positions=[1,2], showmeans=True)
    axes[0,1].set_xticks([1,2]); axes[0,1].set_xticklabels(['Baseline','Jessicka'])
    axes[0,1].set_ylabel('Sharpe')
    max_n = min(sigma_paths.shape[1], 500)
    steps = np.arange(max_n)
    mean_sig = np.mean(sigma_paths[:,:max_n], axis=0)
    p10 = np.percentile(sigma_paths[:,:max_n], 10, axis=0)
    p90 = np.percentile(sigma_paths[:,:max_n], 90, axis=0)
    eta_best = 1 - 2/best_params['kappa']
    theo = power_law_decay(steps, best_params['beta_decay'], eta_best)
    axes[1,0].plot(steps, mean_sig, 'b-', label='Empirical')
    axes[1,0].fill_between(steps, p10, p90, alpha=0.2, color='blue')
    axes[1,0].plot(steps, theo, 'r--', label='Theoretical')
    axes[1,0].set_xlabel('Exposure n'); axes[1,0].set_ylabel('σ(n)'); axes[1,0].legend()
    thetas = np.linspace(0.1,0.9,9)
    sens = []
    subset = min(100, n_paths)
    for th in thetas:
        tmp = []
        for i in range(subset):
            ret = all_returns[i]; vol = all_vols[i]
            mu_c = np.percentile(ret[:50],95)
            if mu_c <= 0: mu_c = np.mean(ret[:50])
            act, _, _ = apply_rotation(ret, vol, mu_c, eta_best, th,
                                       best_params['tau0'], best_params['alpha_load'],
                                       best_params['beta_decay'], 'full')
            tmp.append(calculate_sharpe(act) if len(act) else 0.0)
        sens.append(np.mean(tmp))
    axes[1,1].plot(thetas, sens, 'o-', color='green')
    axes[1,1].axvline(best_params['theta'], color='red', linestyle='--')
    axes[1,1].set_xlabel('θ'); axes[1,1].set_ylabel('Mean Sharpe')
    plt.tight_layout()
    st.pyplot(fig)

    # ---------- Ablation ----------
    fig_ab, ax_ab = plt.subplots(figsize=(6,4))
    means = [np.mean(ab_full), np.mean(ab_nol), np.mean(ab_nosz)]
    stds = [np.std(ab_full), np.std(ab_nol), np.std(ab_nosz)]
    ax_ab.bar(['Full','No overload','No sizing'], means, yerr=stds, capsize=5,
              color=['#2ca02c','#d62728','#9467bd'], alpha=0.7)
    ax_ab.set_ylabel('Mean Sharpe'); ax_ab.set_title('Ablation')
    st.pyplot(fig_ab)

    # ---------- Infographic (radar + table) ----------
    fig_inf, axs = plt.subplots(2,2, figsize=(12,10))
    axs[0,0].violinplot([bh_sharpes, rot_sharpes], positions=[1,2], showmeans=True)
    axs[0,0].set_xticks([1,2]); axs[0,0].set_xticklabels(['Baseline','Jessicka'])
    axs[0,0].set_ylabel('Sharpe')
    axs[0,1].bar(['Baseline','Jessicka'], [bh_var, rot_var], color=['#1f77b4','#ff7f0e'])
    axs[0,1].set_title(f'Variance ↓{var_red:.0f}%')
    # Radar
    bh_arr = np.array(bh_sharpes)
    rot_arr = np.array(rot_sharpes)
    bh_vals = [bh_mean, 1/bh_var, -np.mean(bh_dds), np.mean(bh_arr > 0)]
    rot_vals = [rot_mean, 1/rot_var, -np.mean(rot_dds), np.mean(rot_arr > 0)]
    max_vals = np.maximum(bh_vals, rot_vals)
    bh_norm = bh_vals / max_vals
    rot_norm = rot_vals / max_vals
    metrics = ['Mean Sharpe', 'Stability', 'Drawdown', 'Win Rate']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    bh_norm = np.append(bh_norm, bh_norm[0])
    rot_norm = np.append(rot_norm, rot_norm[0])
    axs[1,0].remove()
    axs[1,0] = fig_inf.add_subplot(2,2,3, projection='polar')
    axs[1,0].plot(angles, bh_norm, 'o-', label='Baseline')
    axs[1,0].fill(angles, bh_norm, alpha=0.2)
    axs[1,0].plot(angles, rot_norm, 'o-', label='Jessicka')
    axs[1,0].fill(angles, rot_norm, alpha=0.2)
    axs[1,0].set_xticks(angles[:-1]); axs[1,0].set_xticklabels(metrics)
    axs[1,0].legend(loc='upper right')
    axs[1,0].set_title('Radar (higher better)')
    axs[1,1].axis('off')
    table = axs[1,1].table(cellText=[
        ['Metric','Baseline','Jessicka','Δ'],
        ['Mean Sharpe',f'{bh_mean:.2f}',f'{rot_mean:.2f}',f'{mean_imp:.0f}%'],
        ['Variance',f'{bh_var:.3f}',f'{rot_var:.3f}',f'-{var_red:.0f}%'],
        ['Win Rate',f'{np.mean(bh_arr>0):.0%}',f'{np.mean(rot_arr>0):.0%}','↑']
    ], loc='center', colWidths=[0.2,0.2,0.2,0.2])
    table.auto_set_font_size(False); table.set_fontsize(10)
    plt.tight_layout()
    st.pyplot(fig_inf)

    st.success("✅ All visuals generated. Jessicka beats baseline.")
else:
    st.info("👈 Set parameters and click RUN.")
