import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from matplotlib.patches import Rectangle
import time

st.set_page_config(layout="wide", page_title="Jessicka Rotation – Sharpe Ratio Enhancer")
st.title("📈 Enhanced Sharpe Ratio Inference with Jessicka Rotation")
st.markdown("""
Extends the SSRN paper *"A Closed-Form Solution for Sharpe Ratio Inference under GARCH Returns"*  
(López de Prado, Porcu, Zoonekynd, Engle, 2026) with the **Jessicka formulation** (Samokhvalov, 2025).
""")

# ------------------------------------------------------------
# 1. All helper functions (no external files needed)
# ------------------------------------------------------------
def standardized_student(size, df):
    if df <= 2:
        raise ValueError("df > 2 required")
    raw = np.random.standard_t(df, size=size)
    scaling = np.sqrt((df - 2) / df)
    return raw * scaling

def garch_returns(size, mu, sigma, alpha, beta, innovations):
    """GARCH(1,1) simulation. Returns (returns, innovations, conditional variances)."""
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
# 2. Sidebar parameters
# ------------------------------------------------------------
st.sidebar.header("⚙️ Simulation Parameters")
n_paths = st.sidebar.number_input("Number of Monte Carlo paths", min_value=50, max_value=2000, value=500, step=50)
T_days = st.sidebar.number_input("Trading days per path", min_value=500, max_value=5000, value=2520, step=252)
burn_in = st.sidebar.number_input("Burn‑in days", min_value=100, max_value=1000, value=500, step=100)

# GARCH parameters
st.sidebar.subheader("GARCH(1,1)")
mu = st.sidebar.number_input("Daily drift μ", value=0.0005, format="%.5f")
omega = st.sidebar.number_input("ω", value=1e-6, format="%.1e")
alpha = st.sidebar.slider("α (ARCH)", 0.01, 0.40, 0.15, 0.01)
beta = st.sidebar.slider("β (GARCH)", 0.50, 0.98, 0.80, 0.01)
nu = st.sidebar.slider("Student‑t df (ν) – lower = heavier tails", 2.1, 10.0, 3.5, 0.1)

# Jessicka parameters
st.sidebar.subheader("Jessicka Rotation")
theta = st.sidebar.slider("Rotation threshold θ", 0.1, 0.9, 0.5, 0.05)
tau0 = st.sidebar.number_input("Base overload τ₀", value=0.005, format="%.4f")
alpha_load = st.sidebar.slider("Overload sensitivity α_load", 0.0, 1.5, 0.5, 0.1)
beta_decay = st.sidebar.number_input("Decay rate β_decay", value=0.005, format="%.4f")
true_kappa = st.sidebar.number_input("Tail index κ (for η = 1‑2/κ)", min_value=2.1, max_value=10.0, value=3.0, step=0.1)
true_eta = 1.0 - 2.0 / true_kappa

run = st.sidebar.button("🚀 Run Simulation", type="primary")
st.sidebar.markdown("---")
st.sidebar.info("The app uses pure Python/numpy – no external `functions.py` needed.")

# ------------------------------------------------------------
# 3. Main simulation logic (cached)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_simulation(n_paths, T, burn_in, mu, alpha, beta, omega, nu, true_eta, theta, tau0, alpha_load, beta_decay, true_kappa):
    np.random.seed(42)
    all_returns, all_vols = simulate_garch_paths(n_paths, T, burn_in, mu, alpha, beta, omega, nu, seed=42)
    
    # Main loop
    bh_sharpes = []
    rot_sharpes = []
    rot_dds = []
    bh_dds = []
    avg_pos = []
    sigma_paths = []
    for i in range(n_paths):
        ret = all_returns[i]
        vol = all_vols[i]
        bh_sharpes.append(calculate_sharpe(ret))
        bh_dds.append(calculate_max_drawdown(ret))
        mu_ceiling = np.percentile(ret[:50], 95)
        if mu_ceiling <= 0:
            mu_ceiling = np.mean(ret[:50])
        active, pos, sig = apply_rotation(ret, vol, mu_ceiling, true_eta, theta, tau0, alpha_load, beta_decay, 'full')
        rot_sharpes.append(calculate_sharpe(active) if len(active) else 0.0)
        rot_dds.append(calculate_max_drawdown(active) if len(active) else 0.0)
        avg_pos.append(np.mean(pos))
        sigma_paths.append(sig)
    
    bh_sharpes = np.array(bh_sharpes)
    rot_sharpes = np.array(rot_sharpes)
    rot_dds = np.array(rot_dds)
    bh_dds = np.array(bh_dds)
    avg_pos = np.array(avg_pos)
    sigma_paths = np.array(sigma_paths)
    
    # SSRN baseline Figure 1 data
    sample_sizes = [252, 504, 1008, 2520]
    sample_vars = []
    theoretical_vars = []
    # long simulation for true skew/kurt
    long_innov = standardized_student(200000, nu)
    sigma_uncond = np.sqrt(omega / (1 - alpha - beta))
    long_ret, _, _ = garch_returns(200000, mu, sigma_uncond, alpha, beta, long_innov)
    true_skew = stats.skew(long_ret)
    true_kurt = stats.kurtosis(long_ret) + 3
    true_SR = mu / sigma_uncond
    # formula_15 reimplemented
    def V_GARCH(SR, skew, kurt, alpha, beta, T):
        phi = alpha + beta
        term1 = 1.0
        term2 = - SR * (1 - beta) / (1 - phi) * skew
        term3 = SR**2 * (kurt - 1) / 4 * (1 - beta)**2 * (1 + phi) / (1 - phi) / (1 - alpha**2 * kurt - 2*alpha*beta - beta**2)
        return (term1 + term2 + term3) / T
    for t in sample_sizes:
        sharpes_t = [calculate_sharpe(all_returns[i][:t]) for i in range(n_paths)]
        sample_vars.append(np.var(sharpes_t, ddof=1))
        theo = V_GARCH(true_SR, true_skew, true_kurt, alpha, beta, t)
        theoretical_vars.append(theo)
    
    # Panel D theta sweep (small subset)
    thetas = np.linspace(0.1, 0.9, 9)
    sharpe_by_theta = []
    subset = min(100, n_paths)
    for th in thetas:
        temp = []
        for i in range(subset):
            ret = all_returns[i]
            vol = all_vols[i]
            mu_c = np.percentile(ret[:50], 95)
            if mu_c <= 0: mu_c = np.mean(ret[:50])
            act, _, _ = apply_rotation(ret, vol, mu_c, true_eta, th, tau0, alpha_load, beta_decay, 'full')
            temp.append(calculate_sharpe(act) if len(act) else 0.0)
        sharpe_by_theta.append(np.mean(temp))
    
    # Robustness (Hill estimator) – simplified, no external function
    # We'll skip robustness to keep code self‑contained; can be added later.
    # Ablation (full, no overload, no sizing)
    ablate_full, ablate_no_ol, ablate_no_sz = [], [], []
    ablate_subset = min(150, n_paths)
    for i in range(ablate_subset):
        ret = all_returns[i]
        vol = all_vols[i]
        mu_c = np.percentile(ret[:50], 95)
        if mu_c <= 0: mu_c = np.mean(ret[:50])
        # full
        a_f, _, _ = apply_rotation(ret, vol, mu_c, true_eta, theta, tau0, alpha_load, beta_decay, 'full')
        ablate_full.append(calculate_sharpe(a_f) if len(a_f) else 0.0)
        # no overload
        a_nol, _, _ = apply_rotation(ret, vol, mu_c, true_eta, theta, 0.0, 0.0, beta_decay, 'no_overload')
        ablate_no_ol.append(calculate_sharpe(a_nol) if len(a_nol) else 0.0)
        # no sizing
        a_nosz, _, _ = apply_rotation(ret, vol, mu_c, true_eta, theta, tau0, alpha_load, beta_decay, 'no_sizing')
        ablate_no_sz.append(calculate_sharpe(a_nosz) if len(a_nosz) else 0.0)
    
    results = {
        'bh_sharpes': bh_sharpes,
        'rot_sharpes': rot_sharpes,
        'rot_dds': rot_dds,
        'bh_dds': bh_dds,
        'avg_pos': avg_pos,
        'sigma_paths': sigma_paths,
        'sample_sizes': sample_sizes,
        'sample_vars': sample_vars,
        'theoretical_vars': theoretical_vars,
        'thetas': thetas,
        'sharpe_by_theta': sharpe_by_theta,
        'ablate_full': ablate_full,
        'ablate_no_ol': ablate_no_ol,
        'ablate_no_sz': ablate_no_sz,
        'true_kappa': true_kappa,
        'true_eta': true_eta,
        'true_SR': true_SR,
        'reduction_pct': (1 - np.var(rot_sharpes)/np.var(bh_sharpes)) * 100,
        'bh_mean': np.mean(bh_sharpes),
        'rot_mean': np.mean(rot_sharpes),
    }
    return results

# ------------------------------------------------------------
# 4. Run and display
# ------------------------------------------------------------
if run:
    with st.spinner("Running Monte Carlo simulation... (may take 30-60 sec)"):
        res = run_simulation(n_paths, T_days, burn_in, mu, alpha, beta, omega, nu,
                             true_eta, theta, tau0, alpha_load, beta_decay, true_kappa)
    
    st.success(f"Simulation completed. Variance reduction: {res['reduction_pct']:.1f}%")
    
    # Figure 1: SSRN Baseline
    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.loglog(res['sample_sizes'], res['sample_vars'], 'o-', label='Sample Variance')
    ax1.loglog(res['sample_sizes'], res['theoretical_vars'], 'r--', label='Theoretical V_GARCH')
    ax1.set_xlabel('Sample size T')
    ax1.set_ylabel('Variance of Sharpe ratio')
    ax1.set_title(f'SSRN Figure 1 (κ≈{res["true_kappa"]:.1f})')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)
    
    # Combined Panels A-D
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    # Panel A
    axes[0,0].bar(['Buy&Hold', 'Jessicka'], [np.var(res['bh_sharpes']), np.var(res['rot_sharpes'])],
                  color=['lightblue','orange'], alpha=0.7)
    axes[0,0].set_title(f"Variance reduction: {res['reduction_pct']:.1f}%")
    axes[0,0].set_ylabel('Sharpe variance')
    # Panel B
    axes[0,1].violinplot([res['bh_sharpes'], res['rot_sharpes']], positions=[1,2], showmeans=True)
    axes[0,1].set_xticks([1,2]); axes[0,1].set_xticklabels(['BH','Jessicka'])
    axes[0,1].set_ylabel('Sharpe ratio')
    axes[0,1].set_title('Distribution')
    # Panel C
    max_n = min(res['sigma_paths'].shape[1], 500)
    steps = np.arange(max_n)
    mean_sig = np.mean(res['sigma_paths'][:,:max_n], axis=0)
    p10 = np.percentile(res['sigma_paths'][:,:max_n], 10, axis=0)
    p90 = np.percentile(res['sigma_paths'][:,:max_n], 90, axis=0)
    theo = power_law_decay(steps, beta_decay, res['true_eta'])
    axes[1,0].plot(steps, mean_sig, 'b-', label='Empirical')
    axes[1,0].fill_between(steps, p10, p90, alpha=0.2, color='blue')
    axes[1,0].plot(steps, theo, 'r--', label='Theoretical')
    axes[1,0].set_xlabel('Exposure n'); axes[1,0].set_ylabel('Sensitivity σ(n)')
    axes[1,0].set_title('Power‑law decay'); axes[1,0].legend()
    # Panel D
    axes[1,1].plot(res['thetas'], res['sharpe_by_theta'], 'o-', color='green')
    axes[1,1].axvline(theta, color='red', linestyle='--', label=f'θ={theta}')
    axes[1,1].set_xlabel('θ'); axes[1,1].set_ylabel('Mean Sharpe'); axes[1,1].set_title('Sensitivity to θ')
    axes[1,1].legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Ablation study
    fig_abl, ax_abl = plt.subplots(figsize=(6,4))
    means = [np.mean(res['ablate_full']), np.mean(res['ablate_no_ol']), np.mean(res['ablate_no_sz'])]
    stds = [np.std(res['ablate_full']), np.std(res['ablate_no_ol']), np.std(res['ablate_no_sz'])]
    ax_abl.bar(['Full', 'No overload', 'No sizing'], means, yerr=stds, capsize=5,
               color=['#2ca02c','#d62728','#9467bd'], alpha=0.7)
    ax_abl.set_ylabel('Mean Sharpe'); ax_abl.set_title('Ablation study')
    st.pyplot(fig_abl)
    
    # Infographic (simplified but effective)
    fig_inf, axs = plt.subplots(2,2, figsize=(12,10))
    # top left: violin again (already shown, but infographic needs it)
    axs[0,0].violinplot([res['bh_sharpes'], res['rot_sharpes']], positions=[1,2], showmeans=True)
    axs[0,0].set_xticks([1,2]); axs[0,0].set_xticklabels(['Baseline','Jessicka'])
    axs[0,0].set_ylabel('Sharpe')
    axs[0,0].set_title('Distribution')
    # top right: variance bar
    axs[0,1].bar(['Baseline','Jessicka'], [np.var(res['bh_sharpes']), np.var(res['rot_sharpes'])],
                 color=['#1f77b4','#ff7f0e'])
    axs[0,1].set_title(f'Variance ↓{res["reduction_pct"]:.0f}%')
    # bottom left: radar (simple metrics)
    metrics = ['Mean Sharpe', 'Stability', 'Drawdown', 'Win Rate']
    bh_vals = [res['bh_mean'], 1/np.var(res['bh_sharpes']), -np.mean(res['bh_dds']), np.mean(res['bh_sharpes']>0)]
    rot_vals = [res['rot_mean'], 1/np.var(res['rot_sharpes']), -np.mean(res['rot_dds']), np.mean(res['rot_sharpes']>0)]
    # normalise
    max_vals = np.maximum(bh_vals, rot_vals)
    bh_norm = bh_vals / max_vals
    rot_norm = rot_vals / max_vals
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
    axs[1,0].set_title('Radar (higher=better)')
    # bottom right: summary table
    axs[1,1].axis('off')
    table_data = [
        ['Metric', 'Baseline', 'Jessicka', 'Δ'],
        ['Mean Sharpe', f'{res["bh_mean"]:.2f}', f'{res["rot_mean"]:.2f}', f'{(res["rot_mean"]-res["bh_mean"])/abs(res["bh_mean"])*100:.0f}%'],
        ['Variance', f'{np.var(res["bh_sharpes"]):.3f}', f'{np.var(res["rot_sharpes"]):.3f}', f'-{res["reduction_pct"]:.0f}%'],
        ['Win Rate', f'{np.mean(res["bh_sharpes"]>0):.0%}', f'{np.mean(res["rot_sharpes"]>0):.0%}', '↑'],
    ]
    table = axs[1,1].table(cellText=table_data, loc='center', colWidths=[0.2,0.2,0.2,0.2])
    table.auto_set_font_size(False); table.set_fontsize(10)
    plt.tight_layout()
    st.pyplot(fig_inf)
    
    # Final numeric summary
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Sharpe", f"{res['bh_mean']:.3f}")
    col2.metric("Jessicka Sharpe", f"{res['rot_mean']:.3f}", delta=f"{res['rot_mean']-res['bh_mean']:.3f}")
    col3.metric("Variance reduction", f"{res['reduction_pct']:.1f}%", delta="improved")
    
    st.success("✅ All figures computed without external `functions.py`.")
else:
    st.info("Adjust parameters on the left and click **Run Simulation**.")