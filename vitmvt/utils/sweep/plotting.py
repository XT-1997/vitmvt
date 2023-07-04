"""Sweep plotting functions."""
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np

from vitmvt.utils.sweep.analysis import get_mm_vals

# Global color scheme and fill color
_COLORS, _COLOR_FILL = [], []


def set_plot_style():
    """Sets default plotting styles for all plots."""
    plt.rcParams['figure.figsize'] = [3.0, 2]
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.4
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['legend.edgecolor'] = '0.3'
    plt.rcParams['axes.xmargin'] = 0.025
    plt.rcParams['lines.linewidth'] = 1.25
    plt.rcParams['lines.markersize'] = 5.0
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['legend.title_fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7


def set_colors(colors=None):
    """Sets the global color scheme (colors should be a list of rgb float
    values)."""
    global _COLORS
    default_colors = [
        [0.000, 0.447, 0.741],
        [0.850, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
        [0.635, 0.078, 0.184],
        [0.300, 0.300, 0.300],
        [0.600, 0.600, 0.600],
        [1.000, 0.000, 0.000],
    ]
    colors = default_colors if colors is None else colors
    colors, n = np.array(colors), len(colors)
    err_str = 'Invalid colors list: {}'.format(colors)
    assert ((colors >= 0) &
            (colors <= 1)).all() and colors.shape[1] == 3, err_str
    _COLORS = np.tile(colors, (int(np.ceil((10000 / n))), 1)).reshape((-1, 3))


def set_color_fill(color_fill=None):
    """Sets the global color fill (color should be a set of rgb float
    values)."""
    global _COLOR_FILL
    _COLOR_FILL = [0.000, 0.447, 0.741] if color_fill is None else color_fill


def get_color(ind=(), scale=1, dtype=float):
    """Gets color (or colors) referenced by index (or indices)."""
    return np.ndarray.astype(_COLORS[ind] * scale, dtype)


def fig_make(m_rows, n_cols, flatten, **kwargs):
    """Gets figure for plotting with m x n axes."""
    figsize = plt.rcParams['figure.figsize']
    figsize = (figsize[0] * n_cols, figsize[1] * m_rows)
    fig, axes = plt.subplots(
        m_rows, n_cols, figsize=figsize, squeeze=False, **kwargs)
    axes = [ax for axes in axes for ax in axes] if flatten else axes
    return fig, axes


def fig_legend(fig, n_cols, names, colors=None, styles=None, markers=None):
    """Adds legend to figure and tweaks layout (call after fig is done)."""
    n, c, s, m = len(names), colors, styles, markers
    c = c if c else get_color()[:n]
    s = [''] * n if s is None else [s] * n if type(s) == str else s
    m = ['o'] * n if m is None else [m] * n if type(m) == str else m
    n_cols = int(np.ceil(n / np.ceil(n / n_cols)))
    hs = [
        lines.Line2D([0], [0], color=c, ls=s, marker=m)
        for c, s, m in zip(c, s, m)
    ]
    fig.legend(
        hs, names, bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=n_cols)
    fig.tight_layout(pad=0.3, h_pad=1.08, w_pad=1.08)


def plot_edf_mm(sweeps, metric='accuracy_top-1/max', names='candidate result'):
    """Plots error EDF for each sweep."""
    m, n = 1, 1
    fig, axes = fig_make(m, n, True)
    for i, sweep in enumerate(sweeps):
        k = len(sweep)
        errs = sorted(get_mm_vals(sweep, metric))
        edf = np.cumsum(np.ones(k) / k)
        label = '{:3d}|{:.1f}|{:.1f}'.format(k, min(errs), np.mean(errs))
        axes[0].plot(errs, edf, '-', alpha=0.8, c=get_color(i), label=label)
    axes[0].legend(loc='lower right', title=' ' * 10 + 'n|min|mean')
    axes[0].set_xlabel('error')
    axes[0].set_ylabel('cumulative prob.')
    fig_legend(fig, n, names, styles='-', markers='')
    return fig


def plot_trends(sweeps, names, metrics, filters, max_cols=0):
    """Plots metric versus sweep for each metric."""
    n_metrics, xs = len(metrics), range(len(sweeps))
    max_cols = max_cols if max_cols else len(sweeps)
    m = int(np.ceil(n_metrics / max_cols))
    n = min(max_cols, int(np.ceil(n_metrics / m)))
    fig, axes = fig_make(m, n, True, sharex=False, sharey=False)
    [ax.axis('off') for ax in axes[n_metrics::]]
    for ax, metric in zip(axes, metrics):
        # Get values to plot
        vals = [get_mm_vals(sweep, metric) for sweep in sweeps]

        vs_min, vs_max = [min(v) for v in vals], [max(v) for v in vals]
        fs_min, fs_med, fs_max = zip(*[f[metric] for f in filters])
        # Show full range
        ax.plot(xs, vs_min, '-', xs, vs_max, '-', c='0.7')
        ax.fill_between(xs, vs_min, vs_max, alpha=0.05, color=_COLOR_FILL)
        # Show good range
        ax.plot(xs, fs_min, '-', xs, fs_max, '-', c='0.5')
        ax.fill_between(xs, fs_min, fs_max, alpha=0.10, color=_COLOR_FILL)
        # Show best range
        ax.plot(xs, fs_med, '-o', c='k')
        # Show good range with markers
        ax.scatter(xs, fs_min, c=get_color(xs), marker='^', s=80, zorder=10)
        ax.scatter(xs, fs_max, c=get_color(xs), marker='v', s=80, zorder=10)
        # Finalize axis
        ax.set_ylabel(metric)
        ax.set_xticks([])
        ax.set_xlabel('sweep')
    fig_legend(fig, n, names, markers='D')
    return fig


# Set global plot style and colors on import
set_plot_style()
set_colors()
set_color_fill()
