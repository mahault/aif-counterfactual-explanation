"""
Publication figures for the adaptive AVCG paper.

Seven figure types:
1. Drift over time — D_t vs t per drift type
2. Validity degradation — strategies comparison, 3-panel per dataset
3. Pareto frontier — compute vs validity tradeoff
4. Drift-validity correlation — scatter with R^2
5. Ablation: drift measures — KL vs JSD vs accuracy delta
6. Decision rule behavior — D_t and validity with REUSE/RETRAIN regions
7. Affective inference trace — F_t, valence, anxiety
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


# Style constants
STRATEGY_STYLES = {
    'NEVER_RETRAIN':  dict(color='#e74c3c', linestyle='--', marker='x', label='Never Retrain'),
    'ALWAYS_RETRAIN': dict(color='#2ecc71', linestyle='-',  marker='s', label='Always Retrain'),
    'ADAPTIVE':       dict(color='#3498db', linestyle='-',  marker='o', label='Adaptive (Ours)'),
    'FIXED_INTERVAL': dict(color='#f39c12', linestyle=':',  marker='^', label='Fixed Interval (K=5)'),
}

DRIFT_COLORS = {
    'covariate':     '#e74c3c',
    'label_noise':   '#3498db',
    'subpopulation': '#2ecc71',
    'rotation':      '#9b59b6',
}


def _load_results(results_dir: str) -> list:
    """Load all results from directory."""
    fpath = os.path.join(results_dir, 'all_results.json')
    if os.path.exists(fpath):
        with open(fpath) as f:
            return json.load(f)

    # Fallback: load individual files
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('.json') and fname != 'all_results.json':
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))
    return results


def _aggregate_by(results: list, group_keys: list, metric_path: list) -> dict:
    """Group results by keys and extract a metric, averaging over seeds."""
    groups = defaultdict(list)
    for r in results:
        key = tuple(r[k] for k in group_keys)
        # Navigate to metric
        val = r
        for p in metric_path:
            if isinstance(val, dict):
                val = val.get(p, None)
            else:
                val = None
                break
        if val is not None:
            groups[key].append(np.array(val, dtype=float))
    # Average over seeds
    return {k: (np.mean(vs, axis=0), np.std(vs, axis=0))
            for k, vs in groups.items() if vs}


# ---------------------------------------------------------------------------
# Figure 1: Drift over time
# ---------------------------------------------------------------------------

def plot_drift_over_time(results: list, out_dir: str):
    """D_t vs t, one line per drift type, averaged over datasets and seeds."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for drift_type, color in DRIFT_COLORS.items():
        Ds = []
        for r in results:
            if r['drift_type'] == drift_type:
                Ds.append(np.array(r['shared']['D_t'], dtype=float))
        if not Ds:
            continue
        mean_D = np.mean(Ds, axis=0)
        std_D = np.std(Ds, axis=0)
        T = np.arange(len(mean_D))
        ax.plot(T, mean_D, color=color, linewidth=2,
                label=drift_type.replace('_', ' ').title())
        ax.fill_between(T, mean_D - std_D, mean_D + std_D,
                         alpha=0.15, color=color)

    ax.set_xlabel('Timestep t', fontsize=12)
    ax.set_ylabel('Predictive KL Divergence $D_t$', fontsize=12)
    ax.set_title('Drift Accumulation Over Time', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig1_drift_over_time.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig1_drift_over_time.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig1_drift_over_time")


# ---------------------------------------------------------------------------
# Figure 2: Validity degradation (3-panel, one per dataset)
# ---------------------------------------------------------------------------

def plot_validity_degradation(results: list, out_dir: str):
    """Validity vs t, four strategies, one subplot per dataset."""
    datasets = sorted(set(r['dataset'] for r in results))
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        for sname, style in STRATEGY_STYLES.items():
            vals = []
            for r in results:
                if r['dataset'] == ds:
                    v = r['strategies'].get(sname, {}).get('metrics', {}).get('validity')
                    if v is not None:
                        vals.append(np.array(v, dtype=float))
            if not vals:
                continue
            mean_v = np.mean(vals, axis=0)
            std_v = np.std(vals, axis=0)
            T = np.arange(len(mean_v))
            ax.plot(T, mean_v, color=style['color'], linestyle=style['linestyle'],
                    marker=style['marker'], markersize=4, linewidth=1.5,
                    label=style['label'], markevery=2)
            ax.fill_between(T, mean_v - std_v, mean_v + std_v,
                             alpha=0.1, color=style['color'])

        ax.set_xlabel('Timestep t', fontsize=11)
        ax.set_title(ds.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Validity', fontsize=11)
    axes[-1].legend(fontsize=9, loc='lower left')
    fig.suptitle('Counterfactual Validity Under Distribution Shift',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig2_validity_degradation.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig2_validity_degradation.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig2_validity_degradation")


# ---------------------------------------------------------------------------
# Figure 3: Pareto frontier — compute vs validity
# ---------------------------------------------------------------------------

def plot_pareto_frontier(results: list, out_dir: str):
    """Scatter of (total retrains, mean validity) for each strategy."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for sname, style in STRATEGY_STYLES.items():
        retrains = []
        mean_vals = []
        for r in results:
            s = r['strategies'].get(sname)
            if s is None:
                continue
            retrains.append(s['retrain_count'])
            v = s['metrics']['validity']
            # Mean validity over t=1..T (skip t=0)
            mean_vals.append(np.mean(v[1:]))

        if not retrains:
            continue

        # Average over experiments
        ax.scatter(np.mean(retrains), np.mean(mean_vals),
                   color=style['color'], marker=style['marker'],
                   s=150, zorder=5, label=style['label'],
                   edgecolors='black', linewidths=0.5)

        # Show spread
        ax.errorbar(np.mean(retrains), np.mean(mean_vals),
                     xerr=np.std(retrains), yerr=np.std(mean_vals),
                     color=style['color'], capsize=4, linewidth=1, zorder=4)

    ax.set_xlabel('Total Retrains', fontsize=12)
    ax.set_ylabel('Mean Validity', fontsize=12)
    ax.set_title('Compute-Validity Pareto Frontier', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig3_pareto_frontier.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig3_pareto_frontier.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig3_pareto_frontier")


# ---------------------------------------------------------------------------
# Figure 4: Drift-validity correlation
# ---------------------------------------------------------------------------

def plot_drift_validity_correlation(results: list, out_dir: str):
    """Scatter D_t vs validity drop, with R^2."""
    fig, ax = plt.subplots(figsize=(7, 5))

    all_D, all_drop = [], []
    for r in results:
        D = np.array(r['shared']['D_t'][1:], dtype=float)
        # Use NEVER_RETRAIN validity (pure drift effect)
        v = r['strategies']['NEVER_RETRAIN']['metrics']['validity']
        v0 = v[0]
        drops = np.array([v0 - vi for vi in v[1:]], dtype=float)
        min_len = min(len(D), len(drops))
        all_D.extend(D[:min_len].tolist())
        all_drop.extend(drops[:min_len].tolist())

    all_D = np.array(all_D)
    all_drop = np.array(all_drop)

    ax.scatter(all_D, all_drop, alpha=0.3, s=20, color='#3498db')

    # Linear fit
    if len(all_D) > 2 and np.std(all_D) > 1e-10:
        coeffs = np.polyfit(all_D, all_drop, 1)
        fit_x = np.linspace(all_D.min(), all_D.max(), 100)
        fit_y = np.polyval(coeffs, fit_x)
        ax.plot(fit_x, fit_y, 'r-', linewidth=2)

        # R^2
        ss_res = np.sum((all_drop - np.polyval(coeffs, all_D))**2)
        ss_tot = np.sum((all_drop - all_drop.mean())**2)
        r_sq = 1 - ss_res / max(ss_tot, 1e-10)
        ax.text(0.05, 0.95, f'$R^2 = {r_sq:.3f}$',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Predictive KL Divergence $D_t$', fontsize=12)
    ax.set_ylabel('Validity Drop (from $t=0$)', fontsize=12)
    ax.set_title('Drift-Validity Correlation', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig4_drift_validity_correlation.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig4_drift_validity_correlation.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig4_drift_validity_correlation")


# ---------------------------------------------------------------------------
# Figure 5: Ablation — drift measures as validity predictors
# ---------------------------------------------------------------------------

def plot_ablation_drift_measures(results: list, out_dir: str):
    """Compare KL vs JSD vs accuracy_delta vs nll_delta as validity predictors."""
    measures = {
        'mean_kl': ('Predictive KL', '#e74c3c'),
        'mean_jsd': ('Jensen-Shannon', '#3498db'),
        'accuracy_delta': ('Accuracy Delta', '#2ecc71'),
        'nll_delta': ('NLL Delta', '#f39c12'),
        'set_size_ratio': ('Set Size Ratio', '#9b59b6'),
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    r_sq_values = {}
    for measure_key, (label, color) in measures.items():
        all_m, all_drop = [], []
        for r in results:
            drift_results = r['shared'].get('drift_result', [])
            v = r['strategies']['NEVER_RETRAIN']['metrics']['validity']
            v0 = v[0]

            for i, dr in enumerate(drift_results):
                if isinstance(dr, dict) and measure_key in dr:
                    val_drop = v0 - v[i + 1] if i + 1 < len(v) else 0
                    m_val = dr[measure_key]
                    if np.isfinite(m_val) and np.isfinite(val_drop):
                        all_m.append(abs(m_val))  # abs for accuracy_delta
                        all_drop.append(val_drop)

        if len(all_m) < 3:
            continue

        all_m = np.array(all_m)
        all_drop = np.array(all_drop)

        if np.std(all_m) > 1e-10:
            corr = np.corrcoef(all_m, all_drop)[0, 1]
            r_sq = corr**2
        else:
            r_sq = 0.0

        r_sq_values[label] = r_sq

    # Bar plot of R^2 values
    if r_sq_values:
        labels_sorted = sorted(r_sq_values, key=r_sq_values.get, reverse=True)
        values = [r_sq_values[l] for l in labels_sorted]
        colors = []
        for l in labels_sorted:
            for mk, (lbl, c) in measures.items():
                if lbl == l:
                    colors.append(c)
                    break

        bars = ax.bar(range(len(labels_sorted)), values, color=colors,
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(labels_sorted)))
        ax.set_xticklabels(labels_sorted, rotation=15, ha='right', fontsize=10)
        ax.set_ylabel('$R^2$ (Validity Prediction)', fontsize=12)
        ax.set_title('Ablation: Drift Measures as Validity Predictors', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig5_ablation_drift_measures.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig5_ablation_drift_measures.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig5_ablation_drift_measures")


# ---------------------------------------------------------------------------
# Figure 6: Decision rule behavior
# ---------------------------------------------------------------------------

def plot_decision_rule(results: list, out_dir: str):
    """D_t and validity on dual axes, colored REUSE/RETRAIN regions."""
    # Use first result that has ADAPTIVE metrics
    adaptive_results = [r for r in results
                        if 'ADAPTIVE' in r.get('strategies', {})]
    if not adaptive_results:
        print("  No ADAPTIVE results found, skipping fig6")
        return

    r = adaptive_results[0]
    adaptive = r['strategies']['ADAPTIVE']['metrics']

    fig, ax1 = plt.subplots(figsize=(10, 5))

    T = np.arange(len(adaptive['validity']))

    # Validity on left axis
    ax1.plot(T, adaptive['validity'], 'b-o', markersize=4, linewidth=1.5,
             label='Validity (Adaptive)')
    ax1.set_xlabel('Timestep t', fontsize=12)
    ax1.set_ylabel('Validity', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # D_t on right axis
    ax2 = ax1.twinx()
    D_t = adaptive['D_t']
    ax2.plot(T, D_t, 'r-^', markersize=4, linewidth=1.5, alpha=0.7,
             label='$D_t$ (KL Drift)')
    ax2.set_ylabel('Predictive KL Divergence $D_t$', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Color RETRAIN timesteps
    retrained = adaptive.get('retrained', [])
    for i, rt in enumerate(retrained):
        if rt:
            ax1.axvspan(i - 0.4, i + 0.4, alpha=0.15, color='green')

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    retrain_patch = mpatches.Patch(color='green', alpha=0.15, label='RETRAIN')
    ax1.legend(lines1 + lines2 + [retrain_patch],
               labels1 + labels2 + ['RETRAIN'],
               loc='lower left', fontsize=9)

    ax1.set_title(f'Adaptive Decision Rule — {r["dataset"]}, {r["drift_type"]}',
                  fontsize=13)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig6_decision_rule.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig6_decision_rule.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig6_decision_rule")


# ---------------------------------------------------------------------------
# Figure 7: Affective inference trace
# ---------------------------------------------------------------------------

def plot_affective_trace(results: list, out_dir: str):
    """F_t, valence, and anxiety over time. Maps to Joffily-Coricelli Fig. 1."""
    adaptive_results = [r for r in results
                        if 'ADAPTIVE' in r.get('strategies', {})]
    if not adaptive_results:
        print("  No ADAPTIVE results found, skipping fig7")
        return

    r = adaptive_results[0]
    adaptive = r['strategies']['ADAPTIVE']['metrics']

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    retrained = adaptive.get('retrained', [])

    # Panel 1: G_reuse and G_retrain (EFE)
    ax = axes[0]
    G_reuse = adaptive.get('G_reuse', [])
    G_retrain = adaptive.get('G_retrain', [])
    if G_reuse and G_retrain:
        T = np.arange(1, len(G_reuse) + 1)
        ax.plot(T, G_reuse, 'r-o', markersize=3, linewidth=1.5, label='$G(\\pi=REUSE)$')
        ax.plot(T, G_retrain, 'g--', linewidth=1.5, label='$G(\\pi=RETRAIN)$')
        for i, rt in enumerate(retrained[1:], 1):
            if rt:
                ax.axvline(i, color='green', alpha=0.3, linewidth=2)
    ax.set_ylabel('Expected Free Energy', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Affective Inference Trace', fontsize=13)

    # Panel 2: Valence
    ax = axes[1]
    valence = adaptive.get('valence', [])
    if valence:
        val_clean = [v if v is not None else 0.0 for v in valence]
        T = np.arange(1, len(val_clean) + 1)
        colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in val_clean]
        ax.bar(T, val_clean, color=colors, alpha=0.7, width=0.8)
        ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Valence $(-\\Delta F)$', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 3: Anxiety
    ax = axes[2]
    anxiety = adaptive.get('anxiety', [])
    if anxiety:
        T = np.arange(1, len(anxiety) + 1)
        ax.plot(T, anxiety, 'purple', linewidth=1.5, marker='D', markersize=3)
        ax.fill_between(T, 0, anxiety, alpha=0.15, color='purple')
        for i, rt in enumerate(retrained[1:], 1):
            if rt:
                ax.axvline(i, color='green', alpha=0.3, linewidth=2)
    ax.set_xlabel('Timestep t', fontsize=11)
    ax.set_ylabel('Anxiety', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig7_affective_trace.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig7_affective_trace.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig7_affective_trace")


# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------

def generate_all_figures(results_dir: str, out_dir: str = None):
    """Generate all 7 publication figures from results directory."""
    if out_dir is None:
        out_dir = os.path.join(results_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    results = _load_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"Loaded {len(results)} experiment results")
    print(f"Generating figures in {out_dir}...")

    plot_drift_over_time(results, out_dir)
    plot_validity_degradation(results, out_dir)
    plot_pareto_frontier(results, out_dir)
    plot_drift_validity_correlation(results, out_dir)
    plot_ablation_drift_measures(results, out_dir)
    plot_decision_rule(results, out_dir)
    plot_affective_trace(results, out_dir)

    print(f"\nAll figures saved to {out_dir}")
