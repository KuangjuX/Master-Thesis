#!/usr/bin/env python3
"""
AffineGraph IR Expressiveness & Complexity Evaluation
=====================================================
Generates publication-quality figures for comparing AffineGraph IR
against CUDA, CuTe, ThunderKittens, Triton, TileLang, and PyTorch.

Target venue style: OSDI / SOSP / PLDI / OOPSLA

Usage:
    python scripts/generate_ir_evaluation.py

Output:
    figures/evaluations/ir_eval_loc_comparison.pdf
    figures/evaluations/ir_eval_feature_matrix.pdf
    figures/evaluations/ir_eval_pareto.pdf
    figures/evaluations/ir_eval_radar.pdf
    figures/evaluations/ir_eval_code_breakdown.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

# ============================================================
# Global style — matches top-venue paper aesthetics
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'evaluations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Color palette — colorblind-friendly, consistent across figs
# ============================================================
COLORS = {
    'CUDA':             '#d62728',   # red
    'CuTe':             '#ff7f0e',   # orange
    'ThunderKittens':   '#2ca02c',   # green
    'Triton':           '#1f77b4',   # blue
    'PyTorch':          '#9467bd',   # purple
    'TileLang':         '#8c564b',   # brown
    'AffineGraph':      '#e377c2',   # pink (highlight)
}

FRAMEWORKS = list(COLORS.keys())

# ============================================================
# Data — replace with your actual measured values
# ============================================================
# Lines of Code (excluding comments & blank lines)
LOC_DATA = {
    #                  CUDA  CuTe  TK    Triton PyTorch TileLang AffineGraph
    'GEMM':          [ 480,  280,  100,   55,   'N/A',   42,      22 ],
    'FlashAttention-2':[ 1500, 780,  200,  150,  'N/A',  105,      52 ],
    'Fused Softmax':  [ 350,  180,  'N/A', 45,   'N/A',   38,      18 ],
    'LayerNorm':      [ 280,  150,  'N/A', 40,   'N/A',   35,      16 ],
    'Fused GEMM':     [ 900,  520,  'N/A', 200,  'N/A',  150,      65 ],
}

# Relative performance vs cuBLAS (%) — for the Pareto chart
# Use FlashAttention-2 as the representative workload
PERF_DATA = {
    #                 LoC   Perf(% of cuBLAS)
    'CUDA':          (1500,  98),
    'CuTe':          (780,  103),
    'ThunderKittens': (200,  100),
    'Triton':        (150,   92),
    'PyTorch':       ('N/A', 60),
    'TileLang':      (105,  102),
    'AffineGraph':   (52,   108),
}

# Feature support matrix
# 0 = unsupported, 1 = partial / manual, 2 = fully automatic
FEATURES = [
    'Explicit Memory\nHierarchy',
    'Bank Conflict\nModeling',
    'Hierarchical\nTiling',
    'Affine Transform\nAnalysis',
    'Auto\nOptimization',
    'Predicated\nExecution',
    'Tensor Core\nMapping',
    'Pipeline\nScheduling',
    'Multi-Arch\nPortability',
]

FEATURE_MATRIX = np.array([
    # CUDA  CuTe  TK   Triton PyTorch TileLang AffineGraph
    [  1,    2,    2,    0,     0,      2,       2  ],  # Explicit Mem Hierarchy
    [  1,    2,    1,    0,     0,      2,       2  ],  # Bank Conflict
    [  1,    2,    2,    1,     0,      2,       2  ],  # Hierarchical Tiling
    [  0,    1,    0,    0,     0,      1,       2  ],  # Affine Transform
    [  0,    1,    1,    2,     0,      2,       2  ],  # Auto Optimization
    [  1,    1,    1,    2,     0,      2,       2  ],  # Predicated Execution
    [  2,    2,    2,    1,     0,      2,       2  ],  # Tensor Core
    [  1,    2,    2,    1,     0,      2,       2  ],  # Pipeline Scheduling
    [  0,    1,    0,    1,     2,      1,       2  ],  # Multi-Arch Portability
])

# Radar chart scores (0–10 scale)
RADAR_CATEGORIES = [
    'Code\nConciseness',
    'Cognitive\nSimplicity',
    'Memory\nAbstraction',
    'Expressiveness',
    'Auto\nOptimization',
    'Compilation\nEfficiency',
    'Hardware\nPortability',
]

RADAR_DATA = {
    'CUDA':           [2, 2, 3, 4, 1, 3, 2],
    'CuTe':           [4, 4, 7, 6, 4, 4, 3],
    'Triton':         [7, 7, 4, 6, 8, 7, 5],
    'TileLang':       [8, 7, 8, 7, 7, 7, 5],
    'AffineGraph':    [9, 9, 10, 9, 9, 8, 8],
}


# ============================================================
# Figure 1: Grouped bar chart — Lines of Code comparison
# ============================================================
def fig1_loc_comparison():
    """
    Grouped bar chart comparing LoC across frameworks for each algorithm.
    Style reference: OSDI '23 Welder Fig.8, PLDI '19 Triton Fig.5
    """
    algorithms = list(LOC_DATA.keys())
    n_algo = len(algorithms)
    n_fw = len(FRAMEWORKS)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    bar_width = 0.11
    x = np.arange(n_algo)

    for j, fw in enumerate(FRAMEWORKS):
        vals = []
        mask = []
        for algo in algorithms:
            v = LOC_DATA[algo][j]
            if v == 'N/A':
                vals.append(0)
                mask.append(False)
            else:
                vals.append(v)
                mask.append(True)

        positions = x + (j - n_fw / 2 + 0.5) * bar_width
        bars = ax.bar(
            positions, vals, bar_width,
            label=fw, color=COLORS[fw],
            edgecolor='white', linewidth=0.5,
            zorder=3,
        )
        for k, (bar, m) in enumerate(zip(bars, mask)):
            if not m:
                bar.set_height(0)

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontweight='bold')
    ax.set_ylabel('Lines of Code (LoC)', fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(10, 3000)
    ax.legend(
        ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.18),
        frameon=True, fancybox=False, edgecolor='gray',
        columnspacing=1.0, handletextpad=0.4,
    )
    ax.set_axisbelow(True)

    # Add reduction annotations for AffineGraph
    for i, algo in enumerate(algorithms):
        ag_val = LOC_DATA[algo][-1]  # AffineGraph is last
        cuda_val = LOC_DATA[algo][0]  # CUDA is first
        if ag_val != 'N/A' and cuda_val != 'N/A':
            reduction = (1 - ag_val / cuda_val) * 100
            pos_x = x[i] + (n_fw - 1 - n_fw / 2 + 0.5) * bar_width
            ax.annotate(
                f'{reduction:.0f}%$\\downarrow$',
                xy=(pos_x, ag_val),
                xytext=(pos_x + 0.08, ag_val * 0.55),
                fontsize=7, color=COLORS['AffineGraph'],
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['AffineGraph'],
                                lw=0.8),
            )

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'ir_eval_loc_comparison.pdf')
    plt.savefig(out)
    print(f'[OK] Saved {out}')
    plt.close()


# ============================================================
# Figure 2: Feature support heatmap / matrix
# ============================================================
def fig2_feature_matrix():
    """
    Heatmap showing hardware-feature support level per framework.
    Style reference: OOPSLA '22 Exo Table 1, PLDI '23 Halide comparison
    """
    n_features = len(FEATURES)
    n_fw = len(FRAMEWORKS)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Custom colormap: red -> yellow -> green
    cmap = ListedColormap(['#FF6B6B', '#FFD93D', '#6BCB77'])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N)

    im = ax.imshow(FEATURE_MATRIX, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(np.arange(n_fw))
    ax.set_yticks(np.arange(n_features))
    ax.set_xticklabels(FRAMEWORKS, fontweight='bold', rotation=30, ha='right')
    ax.set_yticklabels(FEATURES, fontsize=9)

    # Cell annotations using matplotlib markers for better rendering
    for i in range(n_features):
        for j in range(n_fw):
            val = FEATURE_MATRIX[i, j]
            if val == 0:
                # X mark for unsupported
                ax.plot(j, i, marker='x', markersize=12, markeredgewidth=2.5,
                        color='white', zorder=10)
            elif val == 1:
                # Half-filled circle for partial
                ax.plot(j, i, marker='o', markersize=10, markeredgewidth=1.5,
                        markerfacecolor='none', markeredgecolor='#555555', zorder=10)
                # Add a half fill
                half = mpatches.Wedge((j, i), 0.18, 180, 360, 
                                       fc='#555555', ec='none', zorder=11)
                ax.add_patch(half)
            else:
                # Checkmark for full support
                ax.plot(j, i, marker='o', markersize=10,
                        markerfacecolor='white', markeredgecolor='white',
                        markeredgewidth=1.5, zorder=10)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
                        ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Unsupported', 'Partial /\nManual', 'Fully\nAutomatic'],
                            fontsize=8)

    # Highlight AffineGraph column
    col_idx = FRAMEWORKS.index('AffineGraph')
    rect = plt.Rectangle((col_idx - 0.5, -0.5), 1, n_features,
                          linewidth=2.5, edgecolor='black',
                          facecolor='none', zorder=15)
    ax.add_patch(rect)

    ax.set_title('Hardware Feature Support Matrix', fontweight='bold', pad=12)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'ir_eval_feature_matrix.pdf')
    plt.savefig(out)
    print(f'[OK] Saved {out}')
    plt.close()


# ============================================================
# Figure 3: Pareto — Performance vs Code Complexity
# ============================================================
def fig3_pareto():
    """
    Scatter plot: LoC (x, log) vs Performance (y, % of cuBLAS).
    Bubble size encodes expressiveness. Pareto frontier highlighted.
    Style reference: ASPLOS '24 evaluation, ISCA scatter plots
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Compute expressiveness score = sum of feature matrix column
    expr_scores = FEATURE_MATRIX.sum(axis=0)
    expr_dict = {fw: expr_scores[i] for i, fw in enumerate(FRAMEWORKS)}

    plotted = []
    for fw in FRAMEWORKS:
        loc, perf = PERF_DATA[fw]
        if loc == 'N/A':
            continue
        expr = expr_dict[fw]
        size = (expr / expr_scores.max()) * 600 + 80

        ax.scatter(
            loc, perf, s=size,
            color=COLORS[fw], edgecolors='black', linewidth=1.2,
            alpha=0.85, zorder=5, label=fw,
        )
        # Label positioning
        if fw == 'AffineGraph':
            offset_x, offset_y = -10, 8
            ha = 'right'
        elif fw == 'Triton':
            offset_x, offset_y = 8, -8
            ha = 'left'
        elif fw == 'CUDA':
            offset_x, offset_y = 8, -6
            ha = 'left'
        else:
            offset_x, offset_y = 8, 4
            ha = 'left'

        ax.annotate(
            fw, (loc, perf),
            xytext=(offset_x, offset_y), textcoords='offset points',
            fontsize=8.5, fontweight='bold', color=COLORS[fw],
            ha=ha,
        )
        plotted.append((loc, perf, fw))

    # Pareto frontier (lower LoC + higher perf is better -> upper-left)
    plotted.sort(key=lambda p: p[0])
    pareto_x, pareto_y = [plotted[0][0]], [plotted[0][1]]
    best_perf = plotted[0][1]
    for loc, perf, _ in plotted[1:]:
        if perf >= best_perf:
            pareto_x.append(loc)
            pareto_y.append(perf)
            best_perf = perf

    ax.plot(pareto_x, pareto_y, 'k--', linewidth=1.5, alpha=0.5, zorder=2)
    ax.fill_between(pareto_x, pareto_y, 120, alpha=0.04, color='green', zorder=1)
    ax.text(
        pareto_x[0] * 1.1, 113,
        'Pareto-optimal\nregion', fontsize=8, fontstyle='italic',
        color='darkgreen', alpha=0.7,
    )

    ax.annotate(
        r'$\leftarrow$ Less Code, Higher Perf (Ideal)',
        xy=(60, 110), fontsize=8, fontstyle='italic', color='gray',
    )

    ax.set_xscale('log')
    ax.set_xlabel('Lines of Code (LoC) $\\rightarrow$', fontweight='bold')
    ax.set_ylabel('Relative Performance (% of cuBLAS) $\\rightarrow$', fontweight='bold')
    ax.set_xlim(30, 2500)
    ax.set_ylim(55, 118)
    ax.set_axisbelow(True)

    # Bubble size legend
    for s_val, s_label in [(80, 'Low'), (350, 'Med'), (680, 'High')]:
        ax.scatter([], [], s=s_val, c='gray', alpha=0.4, edgecolors='black',
                   linewidth=0.8, label=f'Expr: {s_label}')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower left', ncol=2,
              frameon=True, fancybox=False, edgecolor='gray',
              fontsize=7.5, columnspacing=0.8)

    ax.set_title('Performance vs. Code Complexity (FlashAttention-2)',
                 fontweight='bold', pad=10)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'ir_eval_pareto.pdf')
    plt.savefig(out)
    print(f'[OK] Saved {out}')
    plt.close()


# ============================================================
# Figure 4: Radar chart — multi-dimensional capability
# ============================================================
def fig4_radar():
    """
    Radar / spider chart comparing frameworks across 7 dimensions.
    Style reference: OOPSLA '23 DSL comparison, PLDI '22 Halide
    """
    categories = RADAR_CATEGORIES
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(projection='polar'))

    radar_frameworks = ['CUDA', 'CuTe', 'Triton', 'TileLang', 'AffineGraph']
    linewidths = [1.5, 1.5, 1.5, 1.5, 2.5]  # AffineGraph thicker
    alphas = [0.05, 0.05, 0.05, 0.05, 0.12]

    for fw, lw, alpha in zip(radar_frameworks, linewidths, alphas):
        values = RADAR_DATA[fw] + [RADAR_DATA[fw][0]]
        ax.plot(angles, values, 'o-', linewidth=lw, markersize=4,
                label=fw, color=COLORS[fw])
        ax.fill(angles, values, alpha=alpha, color=COLORS[fw])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=7, color='gray')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.legend(
        loc='upper right', bbox_to_anchor=(1.35, 1.12),
        frameon=True, fancybox=False, edgecolor='gray',
        fontsize=8.5,
    )

    ax.set_title('Multi-dimensional Capability Comparison',
                 fontweight='bold', pad=25, fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'ir_eval_radar.pdf')
    plt.savefig(out)
    print(f'[OK] Saved {out}')
    plt.close()


# ============================================================
# Figure 5: Stacked normalized bar — code composition
# ============================================================
def fig5_abstraction_breakdown():
    """
    Stacked bar chart showing what percentage of code is spent on
    different concerns: memory management, compute logic, synchronization,
    boilerplate. Demonstrates that AffineGraph minimizes boilerplate.
    Style reference: SOSP '23 system decomposition figures
    """
    frameworks_sub = ['CUDA', 'CuTe', 'Triton', 'TileLang', 'AffineGraph']
    categories = ['Boilerplate /\nSetup', 'Memory\nManagement',
                  'Synchronization', 'Compute\nLogic']

    # Percentage breakdown for FlashAttention-2 implementation
    data = np.array([
        # Boilerplate  MemMgmt  Sync  Compute
        [35,           30,      15,   20],   # CUDA
        [25,           30,      15,   30],   # CuTe
        [15,           20,      10,   55],   # Triton
        [12,           22,       8,   58],   # TileLang
        [ 5,           15,       5,   75],   # AffineGraph
    ])

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(frameworks_sub))
    bar_width = 0.55

    bottom = np.zeros(len(frameworks_sub))
    cat_colors = ['#bdc3c7', '#3498db', '#e67e22', '#2ecc71']

    for i, (cat, color) in enumerate(zip(categories, cat_colors)):
        bars = ax.barh(x, data[:, i], bar_width, left=bottom,
                       label=cat, color=color, edgecolor='white', linewidth=0.5)
        for j, (bar, val) in enumerate(zip(bars, data[:, i])):
            if val >= 10:
                ax.text(bottom[j] + val / 2, j, f'{val}%',
                        ha='center', va='center', fontsize=8,
                        fontweight='bold', color='white')
        bottom += data[:, i]

    ax.set_yticks(x)
    ax.set_yticklabels(frameworks_sub, fontweight='bold')
    ax.set_xlabel('Code Composition (%)', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
              ncol=4, frameon=True, fancybox=False, edgecolor='gray',
              fontsize=8, columnspacing=1.0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.set_title('Code Composition Breakdown (FlashAttention-2)',
                 fontweight='bold', pad=35)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'ir_eval_code_breakdown.pdf')
    plt.savefig(out)
    print(f'[OK] Saved {out}')
    plt.close()


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('='*60)
    print('Generating AffineGraph IR Evaluation Figures')
    print('='*60)

    fig1_loc_comparison()
    fig2_feature_matrix()
    fig3_pareto()
    fig4_radar()
    fig5_abstraction_breakdown()

    print('='*60)
    print(f'All figures saved to: {os.path.abspath(OUTPUT_DIR)}')
    print('='*60)
