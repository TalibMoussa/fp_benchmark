"""
lif_fp_benchmark.py
===================
Leaky Integrate-and-Fire (LIF) neuron simulator.

Sections
--------
1.  Biophysical parameters
2.  Deterministic LIF  – Euler integration, three input currents
3.  Stochastic LIF     – Euler–Maruyama, ensemble of trials
4.  Fokker–Planck      – analytical stationary PDF (Ornstein–Uhlenbeck)
                        + finite-difference numerical time evolution
5.  Benchmark figure   – 4-panel publication-quality plot

Run
---
    python lif_fp_benchmark.py

Requires: numpy, scipy, matplotlib  (all standard in MacTeX / conda envs)

Author : Abdulrahman Moussa  (framework: A. Moussa research proposal 2025)
"""

import os
import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────
# 0.  Reproducibility
# ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────
# 1.  Biophysical Parameters
# ─────────────────────────────────────────────────────────────
TAU_M    = 20e-3          # membrane time constant  [s]
V_REST   = -70e-3         # resting potential       [V]
V_THRESH = -50e-3         # spike threshold         [V]
V_RESET  = -75e-3         # reset potential         [V]
V_SPIKE  =  20e-3         # spike peak (cosmetic)   [V]
R_M      = 10e6           # membrane resistance     [Ω]  (1/g_leak)
T_REF    =  2e-3          # absolute refractory     [s]

# Simulation timing
DT       =  0.05e-3       # Euler step              [s]
T_TOTAL  = 200e-3         # total duration          [s]

# Noise amplitude for stochastic simulations
SIGMA    =  1.5e-3        # [V · s^{-1/2}]  (std of Gaussian white noise)

# Three input currents  [A]
I_VALUES = np.array([1.5e-9, 2.5e-9, 3.5e-9])  # sub-, near-, supra-threshold
I_LABELS = [r'$I = 1.5\;\mathrm{nA}$  (sub-threshold)',
            r'$I = 2.5\;\mathrm{nA}$  (near-threshold)',
            r'$I = 3.5\;\mathrm{nA}$  (supra-threshold)']

# Colour palette (colour-blind friendly)
COLORS   = ['#0077BB', '#EE7733', '#009988']

# ─────────────────────────────────────────────────────────────
# 2.  Deterministic LIF – Euler Integration
# ─────────────────────────────────────────────────────────────

def run_lif_deterministic(I_ext, dt=DT, T=T_TOTAL):
    """
    Euler integration of the deterministic LIF equation:

        τ_m dV/dt = -(V - V_rest) + R_m · I_ext

    Returns
    -------
    t  : (N,) array  – time axis [s]
    V  : (N,) array  – membrane voltage trace [V]
    sp : list        – spike times [s]
    """
    N    = int(T / dt)
    t    = np.linspace(0, T, N)
    V    = np.full(N, V_REST)
    sp   = []
    ref  = 0          # refractory counter [steps]

    for i in range(1, N):
        if ref > 0:
            V[i] = V_RESET
            ref -= 1
            continue

        dV   = (-(V[i-1] - V_REST) + R_M * I_ext) / TAU_M
        V[i] = V[i-1] + dt * dV

        if V[i] >= V_THRESH:
            sp.append(t[i])
            V[i]  = V_SPIKE   # cosmetic spike peak
            V[i]  = V_SPIKE
            ref   = int(T_REF / dt)

    # Replace spike-peak samples with reset for next step (already done above)
    return t, V, sp


def compute_firing_rate(spike_times, T=T_TOTAL, warmup=50e-3):
    """Mean firing rate [Hz] excluding the first `warmup` seconds."""
    valid = [s for s in spike_times if s > warmup]
    return len(valid) / (T - warmup)


# ─────────────────────────────────────────────────────────────
# 3.  Stochastic LIF – Euler–Maruyama
# ─────────────────────────────────────────────────────────────

def run_lif_stochastic(I_ext, n_trials=2000, dt=DT, T=T_TOTAL,
                       sigma=SIGMA, seed_state=rng):
    """
    Euler–Maruyama integration of the stochastic LIF SDE:

        dV = [-(V - V_rest) / τ_m  +  R_m·I/τ_m] dt  +  σ dW_t

    Returns final membrane voltages of all trials (after transient).
    Also returns one example trace for visualisation.
    """
    N        = int(T / dt)
    sqrt_dt  = np.sqrt(dt)

    # --- ensemble (vectorised) ---
    V_all    = np.full(n_trials, V_REST)
    ref_all  = np.zeros(n_trials, dtype=int)

    for i in range(N):
        active        = ref_all == 0
        dV            = (-(V_all - V_REST) / TAU_M
                         + R_M * I_ext / TAU_M) * dt \
                        + sigma * seed_state.standard_normal(n_trials) * sqrt_dt
        V_all[active] = V_all[active] + dV[active]

        # threshold crossing
        fired         = active & (V_all >= V_THRESH)
        V_all[fired]  = V_RESET
        ref_all[fired]= int(T_REF / dt)
        ref_all       = np.maximum(ref_all - 1, 0)

    # --- single example trace ---
    Vtrace = np.full(N, V_REST)
    ref    = 0
    spikes = []
    for i in range(1, N):
        if ref > 0:
            Vtrace[i] = V_RESET; ref -= 1; continue
        dV = (-(Vtrace[i-1] - V_REST) / TAU_M
              + R_M * I_ext / TAU_M) * dt \
             + sigma * seed_state.standard_normal() * sqrt_dt
        Vtrace[i] = Vtrace[i-1] + dV
        if Vtrace[i] >= V_THRESH:
            spikes.append(i * dt)
            Vtrace[i] = V_SPIKE
            ref = int(T_REF / dt)

    t = np.linspace(0, T, N)
    return V_all, t, Vtrace


# ─────────────────────────────────────────────────────────────
# 4.  Fokker–Planck Analysis
# ─────────────────────────────────────────────────────────────

def fp_stationary_analytical(v_grid, I_ext, sigma=SIGMA):
    """
    Analytical stationary distribution of the LIF SDE treated as an
    Ornstein–Uhlenbeck (OU) process (ignoring reset for the sub-threshold
    regime; valid when mean drive stays below threshold).

    The OU process  dV = -(V - μ_∞)/τ_m dt + σ dW_t
    has the Gaussian stationary density:

        p_∞(v) = N( μ_∞,  σ²τ_m/2 )

    where  μ_∞ = V_rest + R_m · I_ext.
    """
    mu_inf  = V_REST + R_M * I_ext          # effective rest
    var_inf = (sigma**2 * TAU_M) / 2        # stationary variance
    return norm.pdf(v_grid, loc=mu_inf, scale=np.sqrt(var_inf))


def fp_numerical_evolve(I_ext, sigma=SIGMA,
                        v_min=-90e-3, v_max=-40e-3, Nv=300,
                        T_fp=0.5, dt_fp=0.1e-3):
    """
    Finite-difference Crank–Nicolson solution of the Fokker–Planck PDE:

        ∂p/∂t = -∂/∂v[μ(v)·p] + (σ²/2)·∂²p/∂v²

    with absorbing boundary at v = V_thresh and reflecting at v = v_min.
    After each absorption event the probability mass is re-injected at V_reset
    (simplified reset boundary condition).

    Returns v_grid, p_final  (normalised).
    """
    dv     = (v_max - v_min) / (Nv - 1)
    v      = np.linspace(v_min, v_max, Nv)

    # drift at each grid point
    mu_v   = (-(v - V_REST) + R_M * I_ext) / TAU_M

    D      = sigma**2 / 2                  # diffusion coefficient
    r      = D * dt_fp / dv**2            # diffusion number
    alpha  = mu_v * dt_fp / (2 * dv)      # advection number (centered)

    # Initialise as narrow Gaussian near V_rest
    mu0    = V_REST + R_M * I_ext * 0.1
    p      = norm.pdf(v, mu0, 3e-3)
    p     /= p.sum() * dv

    Nt     = int(T_fp / dt_fp)

    # Pre-build tridiagonal Crank–Nicolson matrices (constant coefficients)
    # LHS:  (I - dt/2 · L) p^{n+1} = (I + dt/2 · L) p^n
    lower  = -(r - alpha[1:])
    upper  = -(r + alpha[:-1])
    center =  1 + 2*r * np.ones(Nv)

    # Boundary: absorbing at V_thresh (right), reflecting at v_min (left)
    center[0]  = 1;  upper[0]   = 0
    center[-1] = 1;  lower[-1]  = 0

    # Store in banded form for scipy
    ab        = np.zeros((3, Nv))
    ab[0, 1:] = upper
    ab[1, :]  = center
    ab[2,:-1] = lower

    for _ in range(Nt):
        # Explicit (RHS)
        rhs       = p.copy()
        rhs[1:-1] = (p[1:-1]
                     + r * (p[2:] - 2*p[1:-1] + p[:-2])
                     + alpha[1:-1] * (p[2:] - p[:-2]) / 2)

        # Reset BC: probability absorbed at right boundary → inject at reset
        absorbed  = p[-1]
        rhs[-1]   = 0        # absorbing
        rhs[0]    = 0        # reflecting

        p = solve_banded((1, 1), ab, rhs)
        p = np.maximum(p, 0)

        # Re-inject at V_reset
        reset_idx        = np.argmin(np.abs(v - V_RESET))
        p[reset_idx]    += absorbed / dv

        # Normalise
        mass = p.sum() * dv
        if mass > 0:
            p /= mass

    return v, p


# ─────────────────────────────────────────────────────────────
# 5.  Publication-Quality Figure
# ─────────────────────────────────────────────────────────────

def make_figure():
    # ── Run all simulations ──────────────────────────────────
    det_results  = [run_lif_deterministic(I) for I in I_VALUES]
    stoch_final  = []
    stoch_trace  = []
    stoch_t      = None
    for I in I_VALUES:
        V_ens, t, Vtr = run_lif_stochastic(I)
        stoch_final.append(V_ens)
        stoch_trace.append(Vtr)
        stoch_t = t

    # FP solutions (use first two currents — both sub-threshold OU regime)
    fp_v_ana, fp_p_ana = [], []
    fp_v_num, fp_p_num = [], []
    for I in I_VALUES[:2]:
        v_g = np.linspace(-90e-3, -48e-3, 500)
        fp_v_ana.append(v_g)
        fp_p_ana.append(fp_stationary_analytical(v_g, I))

        v_n, p_n = fp_numerical_evolve(I)
        fp_v_num.append(v_n)
        fp_p_num.append(p_n)

    # ── Figure layout ────────────────────────────────────────
    plt.rcParams.update({
        'font.family'        : 'DejaVu Serif',
        'font.size'          : 10,
        'axes.labelsize'     : 11,
        'axes.titlesize'     : 10.5,
        'legend.fontsize'    : 8.5,
        'xtick.direction'    : 'in',
        'ytick.direction'    : 'in',
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'axes.spines.top'    : False,
        'axes.spines.right'  : False,
        'mathtext.fontset'   : 'dejavuserif',
    })

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(3, 2,
                            hspace=0.45, wspace=0.35,
                            left=0.08, right=0.97,
                            top=0.94,  bottom=0.07)

    ax_det  = fig.add_subplot(gs[0, :])       # panel A – full width
    ax_sto  = fig.add_subplot(gs[1, 0])       # panel B – stochastic trace
    ax_ens  = fig.add_subplot(gs[1, 1])       # panel C – ensemble histogram
    ax_fp   = fig.add_subplot(gs[2, 0])       # panel D – FP stationary
    ax_bm   = fig.add_subplot(gs[2, 1])       # panel E – benchmark KL / MSE

    mv = lambda v: v * 1e3    # V → mV
    ms = lambda t: t * 1e3    # s → ms

    # ── Panel A – Deterministic voltage traces ───────────────
    t0, _, _ = det_results[0]
    offset   = [0, 0, 0]      # no offset — overlay with threshold line
    for idx, (t, V, sp) in enumerate(det_results):
        ax_det.plot(ms(t), mv(V), color=COLORS[idx],
                    lw=1.3, alpha=0.9, label=I_LABELS[idx])

    ax_det.axhline(mv(V_THRESH), color='gray', ls='--', lw=0.9,
                   label=r'$V_\mathrm{thresh} = -50\;\mathrm{mV}$')
    ax_det.axhline(mv(V_REST),   color='gray', ls=':',  lw=0.9, alpha=0.6,
                   label=r'$V_\mathrm{rest}  = -70\;\mathrm{mV}$')
    ax_det.set_xlabel('Time  [ms]')
    ax_det.set_ylabel('Membrane Potential  [mV]')
    ax_det.set_title(r'\textbf{A}  –  Deterministic LIF: voltage traces'
                     r'  ($\tau_m = 20\;\mathrm{ms},\;R_m = 10\;\mathrm{M\Omega}$)',
                     loc='left')
    ax_det.legend(loc='upper left', ncol=2, framealpha=0.85)
    ax_det.set_xlim(0, ms(T_TOTAL))
    ax_det.set_ylim(-82, 28)

    # Firing rate annotations
    for idx, (_, _, sp) in enumerate(det_results):
        fr = compute_firing_rate(sp)
        if fr > 0:
            ax_det.text(ms(T_TOTAL) * 0.78,
                        mv(V_REST) + idx * 8 - 5,
                        f'FR = {fr:.1f} Hz',
                        color=COLORS[idx], fontsize=8.5)

    # ── Panel B – Stochastic example trace ───────────────────
    ax_sto.plot(ms(stoch_t), mv(stoch_trace[1]),
                color=COLORS[1], lw=0.8, alpha=0.85)
    ax_sto.axhline(mv(V_THRESH), color='gray', ls='--', lw=0.9)
    ax_sto.axhline(mv(V_REST),   color='gray', ls=':',  lw=0.9, alpha=0.6)
    ax_sto.set_xlabel('Time  [ms]')
    ax_sto.set_ylabel('Membrane Potential  [mV]')
    ax_sto.set_title(r'\textbf{B}  –  Stochastic LIF trace  '
                     r'($I = 2.5\;\mathrm{nA},\;\sigma = 1.5\;\mathrm{mV}\cdot\mathrm{s}^{-1/2}$)',
                     loc='left')
    ax_sto.set_xlim(0, ms(T_TOTAL))

    # ── Panel C – Ensemble histogram ─────────────────────────
    bins = np.linspace(-90e-3, -45e-3, 60)
    for idx in range(3):
        vals = stoch_final[idx]
        # clip to below-threshold region for histogram
        vals = vals[vals < V_THRESH]
        ax_ens.hist(mv(vals), bins=mv(bins), density=True,
                    alpha=0.45, color=COLORS[idx], edgecolor='none',
                    label=f'$I = {I_VALUES[idx]*1e9:.1f}$ nA')

    ax_ens.axvline(mv(V_THRESH), color='gray', ls='--', lw=0.9)
    ax_ens.set_xlabel('Membrane Potential  [mV]')
    ax_ens.set_ylabel('Probability Density  [mV$^{-1}$]')
    ax_ens.set_title(r'\textbf{C}  –  Ensemble distribution  '
                     r'(N = 2000 trials)',
                     loc='left')
    ax_ens.legend(framealpha=0.85)

    # ── Panel D – FP stationary PDF comparison ───────────────
    linestyles = ['-', '--']
    for idx in range(2):
        # Numerical FP
        ax_fp.plot(mv(fp_v_num[idx]), fp_p_num[idx] * 1e-3,
                   color=COLORS[idx], lw=2.2,
                   ls=linestyles[idx],
                   label=f'FP numerical  ($I = {I_VALUES[idx]*1e9:.1f}$ nA)')
        # Analytical OU
        ax_fp.plot(mv(fp_v_ana[idx]), fp_p_ana[idx] * 1e-3,
                   color=COLORS[idx], lw=1.2,
                   ls=':', marker='',
                   label=f'OU analytical ($I = {I_VALUES[idx]*1e9:.1f}$ nA)')

    ax_fp.axvline(mv(V_THRESH), color='gray', ls='--', lw=0.9, alpha=0.7)
    ax_fp.set_xlabel('Membrane Potential  [mV]')
    ax_fp.set_ylabel('Probability Density  [mV$^{-1}$]')
    ax_fp.set_title(r'\textbf{D}  –  FP stationary PDF: numerical vs.\ analytical',
                    loc='left')
    ax_fp.legend(fontsize=7.5, framealpha=0.85, ncol=1)

    # ── Panel E – Benchmark: MSE & KL divergence vs σ ────────
    sigma_vals = np.linspace(0.5e-3, 4e-3, 18)
    mse_arr    = np.zeros_like(sigma_vals)
    kl_arr     = np.zeros_like(sigma_vals)

    I_bench    = I_VALUES[0]        # sub-threshold current
    v_bench    = np.linspace(-90e-3, -52e-3, 300)

    for si, sg in enumerate(sigma_vals):
        p_ana = fp_stationary_analytical(v_bench, I_bench, sigma=sg)

        # MC histogram
        V_mc  = np.full(3000, V_REST)
        ref_m = np.zeros(3000, dtype=int)
        N_bm  = int(T_TOTAL / DT)
        sdt   = np.sqrt(DT)
        for _ in range(N_bm):
            act    = ref_m == 0
            dv_mc  = (-(V_mc - V_REST) / TAU_M
                      + R_M * I_bench / TAU_M) * DT \
                     + sg * rng.standard_normal(3000) * sdt
            V_mc[act]  = V_mc[act] + dv_mc[act]
            fired      = act & (V_mc >= V_THRESH)
            V_mc[fired]= V_RESET
            ref_m[fired]= int(T_REF / DT)
            ref_m      = np.maximum(ref_m - 1, 0)

        counts, edges = np.histogram(V_mc, bins=v_bench, density=True)
        v_mid  = 0.5 * (edges[:-1] + edges[1:])
        p_a_m  = fp_stationary_analytical(v_mid, I_bench, sigma=sg)

        mse_arr[si] = np.mean((counts - p_a_m)**2)

        # KL: sum p_ana * log(p_ana / p_mc)  (smooth both to avoid log(0))
        eps       = 1e-9
        total_mc  = counts.sum()
        total_an  = p_a_m.sum()
        if total_mc < eps or total_an < eps:
            kl_arr[si] = np.nan
            continue
        p_mc_norm = counts  / total_mc
        p_an_norm = p_a_m   / total_an
        # Laplace-smooth MC histogram
        p_mc_s = (p_mc_norm + eps) / (1 + eps * len(p_mc_norm))
        p_an_s = (p_an_norm + eps) / (1 + eps * len(p_an_norm))
        kl_arr[si] = float(np.sum(p_an_s * np.log(p_an_s / p_mc_s)))

    ax2 = ax_bm.twinx()
    l1, = ax_bm.plot(sigma_vals * 1e3, mse_arr,
                     'o-', color='#CC3311', lw=1.8, ms=4,
                     label='MSE  (left axis)')
    l2, = ax2.plot(sigma_vals * 1e3, kl_arr,
                   's--', color='#0077BB', lw=1.8, ms=4,
                   label=r'$D_\mathrm{KL}$  (right axis)')
    ax_bm.set_xlabel(r'Noise amplitude $\sigma$  [mV$\cdot$s$^{-1/2}$]')
    ax_bm.set_ylabel('MSE  [V$^{-2}$]', color='#CC3311')
    ax2.set_ylabel(r'$D_\mathrm{KL}$(analytical $\|$ MC)', color='#0077BB')
    ax_bm.tick_params(axis='y', labelcolor='#CC3311')
    ax2.tick_params(axis='y', labelcolor='#0077BB')
    ax_bm.set_title(r'\textbf{E}  –  Benchmark: MSE \& $D_\mathrm{KL}$'
                    r' (analytical OU vs.\ MC)',
                    loc='left')
    ax_bm.legend(handles=[l1, l2], framealpha=0.85, fontsize=8)

    # ── Shared figure label ──────────────────────────────────
    fig.suptitle(
        'LIF Neuron Simulator: Deterministic · Stochastic (Euler–Maruyama) '
        '· Fokker–Planck Benchmark',
        fontsize=12, fontweight='bold', y=0.975
    )

    here = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(here, 'lif_fp_benchmark.pdf'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(here, 'lif_fp_benchmark.png'),
                dpi=200, bbox_inches='tight')
    print(f'Figure saved to: {here}')


# ─────────────────────────────────────────────────────────────
# 6.  Terminal Summary
# ─────────────────────────────────────────────────────────────

def print_summary():
    print('\n' + '='*60)
    print('  LIF Neuron Simulator — Summary')
    print('='*60)
    print(f'  τ_m  = {TAU_M*1e3:.0f} ms  |  V_rest = {V_REST*1e3:.0f} mV  '
          f'|  V_thr = {V_THRESH*1e3:.0f} mV')
    print(f'  σ    = {SIGMA*1e3:.2f} mV·s⁻¹/²  |  dt = {DT*1e3:.2f} ms  '
          f'|  T = {T_TOTAL*1e3:.0f} ms\n')

    print(f'  {"I (nA)":<12}  {"FR_det (Hz)":<16}  '
          f'{"μ_∞ (mV)":<14}  {"σ_∞ (mV)":<12}')
    print('  ' + '-'*58)
    for I in I_VALUES:
        _, _, sp = run_lif_deterministic(I)
        fr       = compute_firing_rate(sp)
        mu_inf   = (V_REST + R_M * I) * 1e3
        sig_inf  = np.sqrt(SIGMA**2 * TAU_M / 2) * 1e3
        print(f'  {I*1e9:<12.1f}  {fr:<16.1f}  '
              f'{mu_inf:<14.2f}  {sig_inf:<12.3f}')
    print('='*60 + '\n')


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print_summary()
    print('Running simulations …  (FP numerical + MC benchmark may take ~30 s)')
    make_figure()
