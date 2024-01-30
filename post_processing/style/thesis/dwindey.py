import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}, \sisetup{detect-all}, \usepackage{sansmath}, \sansmath'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['font.size'] = 10.5
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams["font.family"] = 'serif'
mpl.rcParams["font.serif"] = ['URW Palladio L']

parameters = {
    'textwidth': 4.58, #inch
    'paperwidth': 5.85, # inch
    'paperheight': 8.30, # inch
    'textheight': 6.96, # inch
}

dw_colors = {}

dw_colors['five'] = {'C0': '#80b1d3',
                  'C1': '#fb8072',
                  'C2': '#8dd3c7',
                  'C3': '#bebada',
                  'C4': '#ffffb3'}


dw_colors['five'] = {'C0': '#1f78b4',
                  'C1': '#ff7f00',
                  'C2': '#33a02c',
                  'C3': '#6a3d9a',
                  'C4': '#e31a1c',
                  'C5': '#fb9a99'}

dw_colors['blue'] = {'C0': '#9ecae1',
                  'C1': '#4292c6',
                  'C2': '#08519c'}

dw_colors['green'] = {'C0': '#a1d99b',
                   'C1': '#006d2c',
                   'C2': '#006d2c'}

dw_colors['red'] = {'C0': '#fc9272',
                 'C1': '#ef3b2c',
                 'C2': '#a50f15'}

dw_colors['black'] = {'C0': '#bdbdbd',
                   'C1': '#737373',
                   'C2': '#252525'}

dw_colors['violet'] = {'C0': '#bcbddc',
                    'C1': '#807dba',
                    'C2': '#54278f'}

dw_colors['orange'] = {'C0': '#fdae6b',
                    'C1': '#f16913',
                    'C2': '#a63603'}

annotation_labels = np.array(['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)'])


def annotate(fig, axes, lbl=None, **kwargs):
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
        if lbl is None:
            lbl = annotation_labels[:len(axes_flat)]
        for ax, lb in zip(axes_flat, lbl):
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            height = bbox.height * 72
            ax.annotate(lb, xy=(0, height - 1), xycoords='axes points', ha='left', va='top', **kwargs)
    else:
        bbox = axes.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        height = bbox.height * 72
        axes.annotate(lbl, xy=(0, height - 1), xycoords='axes points', ha='left', va='top', **kwargs)
