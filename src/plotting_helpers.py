from src import helpers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.autonotebook import tqdm
import contextlib
import os

fontfamily = 'Arial'
fontsize = 20

# Fraction of each categorical slot a bar fills. Module-level so talk/poster
# scripts can slim the bars down without touching the paper figures; 0.8 is
# both seaborn's default and what the published figures were made with.
bar_width = 0.8

# Pin the stacked-bar x limits to the categorical slots, (-0.5, n_categories-0.5).
# Without this a panel holding a single bar autoscales its x axis to that bar, so
# the bar fills ~91% of the panel whatever `bar_width` says and slimming it does
# nothing. Off by default: the published figures were made with the autoscaling.
fix_category_xlim = False

# Multiplier applied to the font sizes these plotting functions set for
# themselves. 1.0 keeps the published sizes; talk/poster scripts raise it so the
# labels stay readable when the figure is projected.
font_scale = 1.0

# Label for the grey "not significantly above zero" segments of the stacked
# bars. None leaves them out of the legend, as in the published figures; set it
# to e.g. 'n.s.' to append a grey swatch after the feature entries.
nonsignificant_legend_label = None

# Publication sizing for the bar/strip and stacked bar plots. figure_width_mm is
# the final saved width (legends and labels included) and figure_fontsize is the
# size of every font in those plots, in pt; when set, line widths are thinned to
# suit the small figure. None for both keeps the published sizing.
figure_width_mm = None
figure_fontsize = None

# Width (mm) of the similarity matrix itself (without its labels) at publication size
similarity_matrix_mm = 29.5

# Height (in) the stacked bar plots are drawn at, for 3 inches of width per localizer
# panel. Lower it to make a wide, short figure; 9 is what the published figures used.
stacked_fig_height = 9


def _fs(default):
    """Font size to use: figure_fontsize when set, otherwise the function's own default."""
    return figure_fontsize if figure_fontsize is not None else default


def _lw(default):
    """Line width to use: thin lines at publication size, otherwise the function's own default."""
    return 0.5 if figure_fontsize is not None else default


def _update_font_rc(default_size):
    """Set the base font for a plot, plus the publication-size settings when figure_fontsize is set."""
    plt.rcParams.update({'font.size': _fs(default_size), 'font.family': fontfamily})
    if figure_fontsize is not None:
        plt.rcParams.update({
            'pdf.fonttype': 42, #keeps text editable in the pdf
            'mathtext.default': 'regular', #set $R^2$ in the regular font instead of DejaVu
            'axes.titlesize': figure_fontsize, 'axes.labelsize': figure_fontsize,
            'xtick.labelsize': figure_fontsize, 'ytick.labelsize': figure_fontsize,
            'legend.fontsize': figure_fontsize, 'legend.title_fontsize': figure_fontsize,
            'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
            'xtick.major.size': 2, 'ytick.major.size': 2,
            'lines.linewidth': 0.75, 'patch.linewidth': 0.5, #error bars and legend swatches
        })


def plain_superscripts(text):
    """Swap mathtext R^2 for the unicode superscript character. Mathtext sets
    superscripts at 70% of the font size (6 pt -> 4.2 pt), below journal minimums."""
    return text.replace('$R^2$', 'R²').replace('$r^2$', 'r²')


def save_figure(fig, filepath, dpi=300, pad_inches=0.02):
    """Save a figure tightly cropped; when figure_width_mm is set, first resize it
    (keeping its aspect ratio) until the saved pdf is exactly figure_width_mm wide.
    Text has a fixed size in points, so this converges in a few passes."""
    if figure_width_mm is None:
        fig.savefig(filepath, bbox_inches='tight', dpi=dpi)
        return
    import io
    import re
    import matplotlib.text
    for text in fig.findobj(matplotlib.text.Text):
        text.set_text(plain_superscripts(text.get_text()))
    target_width = figure_width_mm / 25.4
    for _ in range(8):
        # measure the width of the pdf that savefig actually writes
        buffer = io.BytesIO()
        fig.savefig(buffer, format='pdf', bbox_inches='tight', pad_inches=pad_inches)
        saved_width = float(re.search(rb'/MediaBox\s*\[\s*[\d.]+\s+[\d.]+\s+([\d.]+)', buffer.getvalue()).group(1)) / 72
        difference = target_width - saved_width
        if abs(difference) < 0.002:
            break
        width, height = fig.get_size_inches()
        new_width = width + difference
        fig.set_size_inches(new_width, height * new_width / width)
    fig.savefig(filepath, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi)

# Replacement x tick labels for ROI bars, keyed by region name. Only consulted
# for bars whose hemispheres have been averaged together, where the default
# "<region>\nboth" wastes its second line; regions left out, and any plot that
# still splits left from right, keep the default label.
region_tick_labels = {}


def region_tick_label(hemi_mask):
    """Two-line x tick for a '<hemisphere> <region>' bar."""
    parts = hemi_mask.split(' ')
    hemi, region = parts[0], parts[1]
    default = f"{region}\n{hemi}"
    if hemi == 'both':
        return region_tick_labels.get(region, default)
    return default

# --- Thick ROI outlines -----------------------------------------------------
# Ported from naturalistic-social-asd/src/plotting_helpers.py. nilearn's
# plot_surf_contours draws a one-face line, which disappears on a projected
# slide; these paint a band of mesh faces centred on the ROI boundary instead.
# Cache vertex-vertex adjacency per mesh so repeated calls (across views x ROIs)
# don't rebuild it. Keyed by id(faces_array).
_VERTEX_NEIGHBOR_CACHE = {}


def _vertex_neighbors(faces, n_vertices):
    """Build vertex -> set(neighbor vertex indices) from a triangular face array."""
    key = (id(faces), n_vertices)
    if key in _VERTEX_NEIGHBOR_CACHE:
        return _VERTEX_NEIGHBOR_CACHE[key]
    import numpy as np
    neighbors = [set() for _ in range(n_vertices)]
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    neighbors = [np.fromiter(s, dtype=np.int64) for s in neighbors]
    _VERTEX_NEIGHBOR_CACHE[key] = neighbors
    return neighbors


def _dilate_surface_roi(roi_texture, faces, n_iter):
    """Dilate a binary surface ROI by `n_iter` vertex rings."""
    import numpy as np
    if n_iter <= 0:
        return np.asarray(roi_texture, dtype=bool).copy()
    nbrs = _vertex_neighbors(faces, len(roi_texture))
    current = np.asarray(roi_texture, dtype=bool).copy()
    for _ in range(n_iter):
        active = np.where(current)[0]
        if active.size == 0:
            break
        new_idx = np.unique(np.concatenate([nbrs[v] for v in active]))
        current[new_idx] = True
    return current


def _erode_surface_roi(roi_texture, faces, n_iter):
    """Erode a binary surface ROI by `n_iter` vertex rings (a vertex stays only
    if all its mesh neighbors are also in the ROI)."""
    import numpy as np
    if n_iter <= 0:
        return np.asarray(roi_texture, dtype=bool).copy()
    nbrs = _vertex_neighbors(faces, len(roi_texture))
    current = np.asarray(roi_texture, dtype=bool).copy()
    for _ in range(n_iter):
        if not current.any():
            break
        keep = current.copy()
        for v in np.where(current)[0]:
            if not current[nbrs[v]].all():
                keep[v] = False
        current = keep
    return current


def _thick_boundary_face_indices(roi_texture, faces, thickness):
    """Indices of mesh faces forming a solid band centered on the ROI boundary.
    A face is in the band iff (a) at least one of its vertices is in the ROI
    dilated by `thickness` rings, and (b) at least one of its vertices is NOT
    in the ROI eroded by `thickness` rings. This paints every face between the
    eroded and dilated ROIs — no gaps — while keeping the band centered on the
    original boundary. Bandwidth is ~(2*thickness+1) faces."""
    import numpy as np
    inner = _erode_surface_roi(roi_texture, faces, thickness)
    outer = _dilate_surface_roi(roi_texture, faces, thickness)
    if not outer.any():
        return np.array([], dtype=np.int64)
    in_outer = outer[faces].any(axis=1)
    not_all_inner = ~inner[faces].all(axis=1)
    band = in_outer & not_all_inner
    return np.where(band)[0]


def _paint_boundary_faces(ax, face_indices, color):
    """Recolor specific mesh faces on the Poly3DCollection drawn by
    plot_surf_stat_map — mirrors what nilearn.plot_surf_contours does
    internally for boundary faces, but with a caller-supplied face set."""
    from matplotlib.colors import to_rgba
    if face_indices.size == 0 or not ax.collections:
        return
    coll = ax.collections[0]
    rgba = to_rgba(color, alpha=1.0)
    fc_attr = '_facecolor3d' if hasattr(coll, '_facecolor3d') else '_facecolors3d'
    ec_attr = '_edgecolor3d' if hasattr(coll, '_edgecolor3d') else '_edgecolors3d'
    fc = getattr(coll, fc_attr)
    ec = getattr(coll, ec_attr)
    fc[face_indices] = rgba
    if ec.size == 0:
        coll.set_edgecolor(fc)
        ec = getattr(coll, ec_attr)
    ec[face_indices] = rgba


def get_cmaps():
    from matplotlib.colors import LinearSegmentedColormap
    cmaps = {
    'rainbow':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#AD0D1D-F55464-FC9D13-FFD619-F4FF47-4DFF32-1EAF1B-28C448-25FFF9-20AAFD-2031FF-7F27FF-760DA1
    (0.000, (0.678, 0.051, 0.114)),
    (0.083, (0.961, 0.329, 0.392)),
    (0.167, (0.988, 0.616, 0.075)),
    (0.250, (1.000, 0.839, 0.098)),
    (0.333, (0.957, 1.000, 0.278)),
    (0.417, (0.302, 1.000, 0.196)),
    (0.500, (0.118, 0.686, 0.106)),
    (0.583, (0.157, 0.769, 0.282)),
    (0.667, (0.145, 1.000, 0.976)),
    (0.750, (0.125, 0.667, 0.992)),
    (0.833, (0.125, 0.192, 1.000)),
    (0.917, (0.498, 0.153, 1.000)),
    (1.000, (0.463, 0.051, 0.631)))),
    'rainbow_muted': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#BF6C75-FF97A1-FFC571-FFEC94-F9FFA5-A2FF94-8ACC89-84D996-92FFFC-8AD3FF-A7AEFF-C59DFF-A678B9
    (0.000, (0.749, 0.424, 0.459)),
    (0.083, (1.000, 0.592, 0.631)),
    (0.167, (1.000, 0.773, 0.443)),
    (0.250, (1.000, 0.925, 0.580)),
    (0.333, (0.976, 1.000, 0.647)),
    (0.417, (0.635, 1.000, 0.580)),
    (0.500, (0.541, 0.800, 0.537)),
    (0.583, (0.518, 0.851, 0.588)),
    (0.667, (0.573, 1.000, 0.988)),
    (0.750, (0.541, 0.827, 1.000)),
    (0.833, (0.655, 0.682, 1.000)),
    (0.917, (0.773, 0.616, 1.000)),
    (1.000, (0.651, 0.471, 0.725)))),
    'SimCLR_embedding':LinearSegmentedColormap.from_list('SimCLR_embedding', (
    # Edit this gradient at https://eltos.github.io/gradient/#EDDFF7-3F2352
    (0.000, (0.929, 0.875, 0.969)),
    (1.000, (0.247, 0.137, 0.322)))),
    'SLIP_embedding': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#F5D0DF-791932
    (0.000, (0.961, 0.816, 0.875)),
    (1.000, (0.475, 0.098, 0.196)))),
    'SimCLR_attention': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#FFFDEA-79521A
    (0.000, (1.000, 0.992, 0.918)),
    (1.000, (0.475, 0.322, 0.102)))),
    'SLIP_attention': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#EFF4F3-1A7975
    (0.000, (0.937, 0.957, 0.953)),
    (1.000, (0.102, 0.475, 0.459)))),
    'blue_neg_yellow_pos': LinearSegmentedColormap.from_list('yellow_hot', (
                # Edit this gradient at https://eltos.github.io/gradient/#4C71FF-0025B3-000000-C73A03-FCEB4A
                (0.000, (0.298, 0.443, 1.000)),
                (0.250, (0.000, 0.145, 0.702)),
                (0.500, (0.000, 0.000, 0.000)),
                (0.750, (0.780, 0.227, 0.012)),
                (1.000, (0.988, 0.922, 0.290)))),
    'yellow_hot': LinearSegmentedColormap.from_list('my_gradient', (
        # Edit this gradient at https://eltos.github.io/gradient/#0:000000-51:C73A03-100:FCEB4A
        (0.000, (0.000, 0.000, 0.000)),
        (0.510, (0.780, 0.227, 0.012)),
        (1.000, (0.988, 0.922, 0.290)))),
    'teal_orange': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:FFAC00-19.9:9D7625-50:000000-82:1F6C5F-100:00FFD3
    (0.000, (1.000, 0.675, 0.000)),
    (0.199, (0.616, 0.463, 0.145)),
    (0.500, (0.000, 0.000, 0.000)),
    (0.820, (0.122, 0.424, 0.373)),
    (1.000, (0.000, 1.000, 0.827)))),
    'alexnet': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#E1E9FE-173C92
    (0.000, (0.882, 0.914, 0.996)),
    (1.000, (0.090, 0.235, 0.573)))),
    'sbert': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#FFEADA-E07F2E
    (0.000, (1.000, 0.918, 0.855)),
    (1.000, (0.878, 0.498, 0.180)))),
    'sbert_alexnet': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#E07F2E-000000-173C92
    (0.000, (0.878, 0.498, 0.180)),
    (0.500, (0.000, 0.000, 0.000)),
    (1.000, (0.090, 0.235, 0.573)))),
    'alexnet_sbert': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#173C92-000000-E07F2E
    (0.000, (0.090, 0.235, 0.573)),
    (0.500, (0.000, 0.000, 0.000)),
    (1.000, (0.878, 0.498, 0.180)))),
    'just_purple':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#A28DC6
    (0.000, (0.635, 0.553, 0.776)),
    (1.000, (0.635, 0.553, 0.776)))),
    'audio':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#FEFDFF-97002B
    (0.000, (0.996, 0.992, 1.000)),
    (1.000, (0.592, 0.000, 0.169)))),
    'red':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#FFE0D6-902F31
    (0.000, (1.000, 0.878, 0.839)),
    (1.000, (0.565, 0.184, 0.192)))),
    'orange':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#FFF1D6-A77000
    (0.000, (1.000, 0.945, 0.839)),
    (1.000, (0.655, 0.439, 0.000)))),
    'green':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#D6FFD9-02770A
    (0.000, (0.839, 1.000, 0.851)),
    (1.000, (0.008, 0.467, 0.039)))),
    'blue':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#D6D6FF-010677
    (0.000, (0.839, 0.839, 1.000)),
    (1.000, (0.004, 0.024, 0.467)))),
    'purple':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#E5D6FF-522F90
    (0.000, (0.898, 0.839, 1.000)),
    (1.000, (0.322, 0.184, 0.565)))),
    'SLIP':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#4A34B0-4A34B0
    (0.000, (0.290, 0.204, 0.690)),
    (1.000, (0.290, 0.204, 0.690)))),
    'SimCLR':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#58B9C8-58B9C8
    (0.000, (0.345, 0.725, 0.784)),
    (1.000, (0.345, 0.725, 0.784)))),
    'GPT2':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#810014-810014
    (0.000, (0.506, 0.000, 0.078)),
    (1.000, (0.506, 0.000, 0.078)))),
    'gray_inferno': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:A99F9F-25:5A1B6A-50:AF4354-75:EB9337-100:F9FBAD
    (0.000, (0.663, 0.624, 0.624)),
    (0.250, (0.353, 0.106, 0.416)),
    (0.500, (0.686, 0.263, 0.329)),
    (0.750, (0.922, 0.576, 0.216)),
    (1.000, (0.976, 0.984, 0.678)))),
    'tulip': LinearSegmentedColormap.from_list('my gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#D32D42-F3F761
    (0.000, (0.827, 0.176, 0.259)),
    (1.000, (0.953, 0.969, 0.380)))),
    'tulip_symmetric': LinearSegmentedColormap.from_list('my gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#F3F761-D32D42-F3F761
    (0.000, (0.953, 0.969, 0.380)),
    (0.500, (0.827, 0.176, 0.259)),
    (1.000, (0.953, 0.969, 0.380)))),
    'gray_inferno_symmetric': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#F9FBAD-EB9337-AF4354-5A1B6A-A99F9F-5A1B6A-AF4354-EB9337-F9FBAD
    (0.000, (0.976, 0.984, 0.678)),
    (0.125, (0.922, 0.576, 0.216)),
    (0.250, (0.686, 0.263, 0.329)),
    (0.375, (0.353, 0.106, 0.416)),
    (0.500, (0.663, 0.624, 0.624)),
    (0.625, (0.353, 0.106, 0.416)),
    (0.750, (0.686, 0.263, 0.329)),
    (0.875, (0.922, 0.576, 0.216)),
    (1.000, (0.976, 0.984, 0.678)))),
    'inferno_symmetric': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:F9FBAD-15.4:EB9337-30.6:AF4354-50:5A1B6A-68.1:AF4354-84.5:EB9337-100:F9FBAD
    (0.000, (0.976, 0.984, 0.678)),
    (0.154, (0.922, 0.576, 0.216)),
    (0.306, (0.686, 0.263, 0.329)),
    (0.500, (0.353, 0.106, 0.416)),
    (0.681, (0.686, 0.263, 0.329)),
    (0.845, (0.922, 0.576, 0.216)),
    (1.000, (0.976, 0.984, 0.678)))),
    'inferno': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0.5:5A1B6A-33:AF4354-66:EB9337-100:F9FBAD
    (0.000, (0.353, 0.106, 0.416)),
    (0.005, (0.353, 0.106, 0.416)),
    (0.330, (0.686, 0.263, 0.329)),
    (0.660, (0.922, 0.576, 0.216)),
    (1.000, (0.976, 0.984, 0.678)))),
    'matrix_green': LinearSegmentedColormap.from_list('my_gradient', (
        # Edit this gradient at https://eltos.github.io/gradient/#FFFFFF-376A5F
        (0.000, (1.000, 1.000, 1.000)),
        (1.000, (0.216, 0.416, 0.373))))
    }

    cmaps['GPT2_1sent'] = cmaps['sbert']
    cmaps['GPT2'] =cmaps['red']
    cmaps['GPT2_3sent'] = cmaps['sbert']
    cmaps['GPT2_chunk60'] = cmaps['sbert']
    cmaps['GPT2_chunk80'] = cmaps['sbert']
    cmaps['GPT2_chunk120'] = cmaps['sbert']


    cmaps['GPT2_chunk60_eps0.1'] = cmaps['sbert']
    cmaps['hubert']=cmaps['audio']
    cmaps['hubert_chunk60']=cmaps['audio']
    cmaps['hubert_chunk120']=cmaps['audio']

    cmaps['alexnet_eps0.1'] = cmaps['alexnet']
    cmaps['alexnet_eps0.3'] = cmaps['alexnet']

    cmaps['SimCLR_attention_eps0.1'] = cmaps['alexnet']
    cmaps['SimCLR_attention_eps0.2'] = cmaps['alexnet']
    cmaps['SimCLR_attention_eps0.4'] = cmaps['alexnet']
    cmaps['SimCLR_attention_eps0.3'] = cmaps['alexnet']
    cmaps['SimCLR_attention_eps0.6'] = cmaps['alexnet']

    cmaps['GPT2_1word'] = cmaps['red']
    
    return cmaps
def get_colors_dict():
    models_dict = helpers.get_models_dict()
    cmaps = get_cmaps()
    colors_dict = {
                 None:'white',
                'GPT2':'#810014',
                'GPT2_1sent':'#810014',
                'blank':'white',
                #GREENS
                'social':'#9E9AC8',#'#6b802f',#'limegreen',
                'non-social':'#D3ECCD',
                'num_agents':'#0f4a2e',#'olive',
                'number of agents':'#0f4a2e',#'olive',
                'face': '#277677',#'green',
                #PINKS
                'valence': '#98597c',#'plum',
                'arousal': '#6e3355',#'palevioletred',
                #ORANGES
                'turn_taking':'#c44601',
                'turn taking':'#c44601',
                'mentalization': '#f0aa00',
                'written_text': '#c17400',
                'written text': '#c17400',
                'speaking': '#f57600',#f36611',
                #REDS
                'amplitude': '#6e0d0c', 
                'pitch': '#9a1f11',
                'music': '#ee292f',
                #BLUES
                'indoor_outdoor': '##5667a2',#'aquamarine',
                'indoor outdoor': '##5667a2',#'aquamarine',
                'hue':'#15385c',#'#856798', #xkcd dark lavender
                'pixel': '#2b5c83',#',#xkcd light violet
                'motion':'#92B4D6',#'#952e8f', #xkcd warm purple
                'joint':'green',
                'joint_masked_faces':'gold',
                'joint_masked_random':'grey',
                'joint_masked_faces_blur':'brown'
                 }
    for feature in ['alexnet_eps0.3','SimCLR_attention_eps0.1','SimCLR_attention_eps0.2','SimCLR_attention_eps0.4','SimCLR_attention_eps0.3','SimCLR_attention_eps0.6','hubert','alexnet','alexnet_eps0.1','sbert','GPT2_chunk120','GPT2_chunk60','GPT2_chunk80','GPT2_chunk60_eps0.1','hubert_chunk60','hubert_chunk120']:
        colors_dict[feature]=cmaps[feature](0.9) #get the color for the combined layers, then assign more specific colors to each layer
        for i,layer in enumerate(models_dict[feature]):
            colors_dict[layer] = cmaps[feature](i/len(models_dict[feature])) #normalized by total number of layers
    
    for i,layer in enumerate(models_dict['alexnet']):
        colors_dict['AlexNet layer '+layer[-1]] = cmaps['alexnet'](i/len(models_dict['alexnet'])) #normalized by total number of layers
    for i,layer in enumerate(models_dict['sbert']):
        colors_dict['sBERT layer '+layer.split('layer')[1]] = cmaps['sbert'](i/len(models_dict['sbert'])) #normalized by total number of layers

    
    colors_dict['interact-no_interact'] = colors_dict['social']
    colors_dict['interact&no_interact'] = colors_dict['motion']
    colors_dict['intact-degraded'] = '#D65B06'
    colors_dict['social interaction'] = colors_dict['social']
    colors_dict['language'] = '#D65B06'
    colors_dict['alexnet+motion'] = colors_dict['alexnet']
    colors_dict['sbert+word2vec'] = colors_dict['sbert']

    colors_dict['alexnet_eps0.3'] = colors_dict['alexnet']
    colors_dict['SimCLR_eps0.3'] = colors_dict['alexnet']
    
   
    colors_dict['vislang_gpt2_chunk120_eps0.3'] = 'green'
    colors_dict['joint_chunk120'] = 'green'

    colors_dict['GPT2_chunk120'] = '#D65B06'
    colors_dict['gpt2_chunk120'] = '#D65B06'

    colors_dict['word2vec'] = '#F57600'
    colors_dict['word2vec_chunk120'] = '#F57600'

    colors_dict['hubert'] = '#E8A06D'
    colors_dict['hubert_chunk120'] = colors_dict['hubert']

    
    colors_dict['AlexNet+motion'] = colors_dict['alexnet']
    colors_dict['SimCLR'] = colors_dict['alexnet']
    colors_dict['SimCLR+motion'] = colors_dict['SimCLR']

    colors_dict['HuBERT+word2vec+GPT2'] = colors_dict['GPT2_chunk120']

    
    
    colors_dict['vision'] = colors_dict['alexnet']
    colors_dict['vision_simclr'] = colors_dict['alexnet']
    colors_dict['AlexNet'] = colors_dict['alexnet']
    colors_dict['Image Model'] = colors_dict['alexnet']
    colors_dict['Motion Model'] = colors_dict['motion']
    colors_dict['Speech Model'] = colors_dict['hubert']
    colors_dict['Word Model'] = colors_dict['word2vec']
    colors_dict['Language Model'] = colors_dict['GPT2_chunk120']
    colors_dict['Sentence Model'] = colors_dict['GPT2_chunk120']
    colors_dict['SimCLR attention'] = colors_dict['alexnet']



    colors_dict['alexnet_masked_faces'] = colors_dict['alexnet']
    colors_dict['language'] = colors_dict['GPT2_chunk120']
    
    colors_dict['vision_transformers'] = colors_dict['SimCLR']
    colors_dict['language_transformers'] = colors_dict['GPT2_1sent']
    
    colors_dict['SimCLR+motion'] = colors_dict['SimCLR']
    colors_dict['HuBERT+word2vec+GPT2'] = colors_dict['GPT2_chunk120']
    
    colors_dict['cross-subject encoding'] = 'grey'
    
    return colors_dict
def plot_img_volume(img,filepath,threshold=None,vmin=None,vmax=None,cmap='cold_hot',title=None,symmetric_cbar='auto'):
    import nibabel
    import numpy as np
    from nilearn import plotting

    #NOTE: view_surf simply plots a surface mesh on a brain, no tri averaging like plot_surf_stat_map
    #NOTE: you can't have any NaN values when using view_surf -- it doesn't handle the threshold correctly
    # so, converting all NaN's negative infinity and then turning all negatives to 0 for plotting
    
    display = plotting.plot_glass_brain(
            stat_map_img = img,
            output_file=filepath,
            colorbar=True,
            cmap=cmap,
            threshold=threshold,
            display_mode='lr',#'lyrz',lzr
            vmin=vmin,
            vmax=vmax,
            title=title,
            symmetric_cbar=symmetric_cbar,
            plot_abs=False
            # norm=norm
            ) 
    # view = plotting.view_img(
    #     img, title=title, cut_coords=[36, -27, 66],vmin=0,vmax=0.5, symmetric_cmap=False,opacity=0.5,
    # )
    # view.open_in_browser()
def plot_surface(nii, filename, ROI_niis=[], threshold=None, vmin=None, vmax=None, title=None, cmap='cold_hot', symmetric_cbar='auto',colorbar_label='',views=['lateral','ventral'],ROIs=[],ROI_colors=[],width_mm=90,fontsize=6,png_dpi=None):
    #width_mm and fontsize are the printed panel width and font size (see compose_final_figure_editable)
    #png_dpi: also save a .png of the panel at this dpi (for figures composed from pngs)
    import nilearn.datasets
    from nilearn import surface
    from nilearn import plotting
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import gc

    temp_filename = filename#'/'.join(filename.split('/')[:-1])

    fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')

    # Pre-compute textures once per hemisphere (avoids redundant vol_to_surf calls)
    n_points_to_sample = 50
    cached_textures = {}
    cached_roi_textures = {}
    for hemi in ('left','right'):
        transform_mesh = fsaverage['pial_' + hemi]
        inner_mesh = fsaverage['white_' + hemi]
        cached_textures[hemi] = surface.vol_to_surf(nii, transform_mesh, inner_mesh=inner_mesh, depth=np.linspace(0, 1, n_points_to_sample), interpolation='nearest')
        cached_textures[hemi] = np.nan_to_num(cached_textures[hemi], nan=0)
        # Pre-compute ROI textures per hemisphere
        cached_roi_textures[hemi] = []
        for ROI_nii in ROI_niis:
            roi_texture = surface.vol_to_surf(ROI_nii, transform_mesh, inner_mesh=inner_mesh, depth=np.linspace(0, 1, n_points_to_sample), interpolation='linear')
            roi_texture = (roi_texture > 0.03) * 1
            cached_roi_textures[hemi].append(roi_texture)

    # Derive vmin/vmax from cached textures if not specified
    if vmax is None:
        vmax = max(round(np.nanmax(cached_textures[h]), 2) for h in ('left','right'))
    if vmin is None:
        vmin = min(round(np.nanmin(cached_textures[h]), 2) for h in ('left','right'))

    vmax = np.max([vmax,-vmin])
    if(vmin!=0):
        vmin = -vmax
    if vmax <= vmin: #empty map: nothing is drawn, so show a positive range instead of matplotlib's -0.1 to 0.1 fallback
        vmax = vmin + 0.1
    #### SAVE each brain separately ######
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})

    for (hemi, view) in [(hemi,view) for view in views for hemi in ['left','right']]:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(5, 5))

        plot_mesh = fsaverage['infl_' + hemi]
        bg_map = fsaverage['sulc_' + hemi]

        texture = cached_textures[hemi].copy()

        if vmin is not None:
            texture[texture < vmin] = vmin
        if threshold is not None:
            texture[np.abs(texture) < threshold] = 0

        plotting.plot_surf_stat_map(plot_mesh, texture, hemi=hemi,
                                    view=view, colorbar=False,
                                    threshold=0.00000000001,
                                    bg_map=bg_map,
                                    cmap= get_cmaps().get(cmap, cmap) if isinstance(cmap, str) else cmap,
                                    symmetric_cbar=symmetric_cbar,
                                    # vmin=vmin,
                                    vmax=vmax,
                                    axes=ax,#axes_dict[(hemi, view)],
                                    engine='matplotlib')
        for ind, roi_texture in enumerate(cached_roi_textures[hemi]):
            try:
                plotting.plot_surf_contours(plot_mesh,roi_texture,avg_method='mean',axes=ax,levels=[1],labels=[ROIs[ind]],colors=[ROI_colors[ind]])
            except Exception as e:
                print(filename)
                print(e)
                pass

        if view=='ventral':
            if hemi == 'left':
                ax.view_init(elev=270,azim=180)# Rotate the (left, ventral) view by 180 degrees
            elif hemi == 'right':
                ax.view_init(elev=270,azim=0)
            ax.set_box_aspect((1, 1, 1), zoom=1.3) #zoom in because brains are too small

        plt.savefig(temp_filename+'_'+hemi+'_'+view+'.png', bbox_inches='tight', transparent=True, dpi=300)
        plt.close()
        gc.collect()

    ##crop each separate brain plot
    for (hemi,view) in [(hemi,view) for view in views for hemi in ['left','right']]:
        if(view=='lateral'):
            left=150
            top=240
            width=1100
            height=950
        elif(view=='ventral'):
            left=150
            top=400
            width=1100
            height=775
        elif(view=='medial'):
            left=150
            top=215
            width=1100
            height=950
        crop_image(temp_filename+'_'+hemi+'_'+view+'.png',temp_filename+'_'+hemi+'_'+view+'.png',left,top,width,height)
    
    ### put together all of the cropped images into one plot
    list_images = [temp_filename+'_'+hemi+'_'+view+'.png' for view in views for hemi in ['left','right']]
    #colorbar and title are drawn by matplotlib so their text stays editable in the pdf
    compose_final_figure_editable(filename+'.pdf', list_images, get_cmaps()[cmap.split('_symmetric')[0]], vmin, vmax, colorbar_label=colorbar_label, title=title, width_mm=width_mm, fontsize=fontsize, png_dpi=png_dpi)
    for (hemi,view) in [(hemi,view) for view in views for hemi in ['left','right']]:
            delete_file = temp_filename+'_'+hemi+'_'+view+'.png'
            os.remove(delete_file)
def plot_localizer(self,task,p_threshold=1,vmin=None,vmax=None,symmetric_cbar=True,cmap='yellow_hot',plot_outlines = False):
        import nibabel
        from nilearn.glm import threshold_stats_img
        
        label = 'zscore'
        for contrast in self.localizer_contrasts[task]:
            subject_tqdm = tqdm(self.subjects[task])
            for subject in subject_tqdm:
                subject_tqdm.set_description(contrast + ': ' +subject)
                try:
                    # print(subject)
                    filepath = os.path.join(self.glm_dir,subject, subject+ "_task-"+ task+'_space-'+self.space+ "_run-"+ self.all_runs[task]+ "_measure-"+label + "_contrast-"+contrast+".nii.gz")
                    nii = nibabel.load(filepath)
                    # plot individual subject too
                    ROI_niis = []
                    filename = subject+self.glm_file_label+'_measure-'+label+'_contrast-'+contrast
                    if plot_outlines:
                        glm_file_label_ = '_smoothingfwhm-'+str(self.smoothing_fwhm)
                        file_label_ = subject+glm_file_label_+'_mask-'+'_'.join(self.localizer_masks[contrast])#mask
                        ROI_file = self.out_dir + '/localizer_masks/'+file_label_+'_glm_loc-'+contrast+'_run-all_binary.nii.gz'
                        try:
                            ROI_niis.append(nibabel.load(ROI_file))
                            filename = filename + '_loc-' + contrast
                        except Exception as e:
                            print(e)
                    ind_filepath = os.path.join(self.figure_dir,'glm_zscores',filename)
                    cmap_ = cmap
                    threshold_dict = {0.05: 1.72,
                                  0.01: 2.5,
                                  0.001: 3.6,
                                  1:None}
                    thresholded_map, threshold_corrected = threshold_stats_img(
                        nii, alpha=threshold_dict[p_threshold], height_control=None,two_sided=False #only care about positive direction?
                            )
                    plot_surface(thresholded_map,ind_filepath,threshold=0.01,title='',symmetric_cbar=symmetric_cbar,vmin=vmin,vmax=vmax,cmap=cmap_,colorbar_label='z-score', ROIs = [contrast], ROI_niis = ROI_niis, ROI_colors = ['white'])
                except Exception as e:
                    print(e)
                    pass
def save_colorbar(cmap, vmin, vmax, filename,colorbar_label,make_cmap=True):
    import matplotlib.pyplot as plt
    import numpy as np
    
    colorbar_height = 3.5
    fig, ax = plt.subplots(figsize=(0.5, colorbar_height))
    fig.subplots_adjust(right=0.5)
    
    if(make_cmap):
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    else:
        sm=plt.cm.ScalarMappable(cmap=cmap)

    
    cbar = plt.colorbar(sm, cax=ax,aspect=50)
    cbar.set_label(colorbar_label)
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.set_ylim(vmin, vmax)
    
    plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
def compose_final_figure_editable(output_filename, cropped_images, cmap, vmin, vmax, colorbar_label='', title=None, num_columns=2, dpi=300, width_mm=90, fontsize=6, title_weight='normal', png_dpi=None):
    #same grid layout as compose_final_figure, but built in matplotlib instead of PIL so that the
    #colorbar, its label, and the title are saved as editable text (the brain renders stay raster).
    #The panel is drawn at its final printed width (width_mm) so fontsize is the printed font size.
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    from PIL import Image

    images = [np.asarray(Image.open(img_path)) for img_path in cropped_images]
    num_rows = math.ceil(len(images) / num_columns)

    # Compute row heights and column widths (pixels)
    row_heights = [0] * num_rows
    col_widths = [0] * num_columns
    for index, img in enumerate(images):
        row, col = divmod(index, num_columns)
        row_heights[row] = max(row_heights[row], img.shape[0])
        col_widths[col] = max(col_widths[col], img.shape[1])

    # Sizes in inches: the colorbar strip is fixed, the brains are scaled to fill the rest of width_mm
    fig_width = width_mm / 25.4
    colorbar_width = 0.5 if colorbar_label != '' else 0 #space for label, bar and ticks
    grid_width = fig_width - colorbar_width
    in_per_px = grid_width / sum(col_widths)
    grid_height = sum(row_heights) * in_per_px
    title_height = 2 * fontsize / 72 if title else 0
    fig_height = grid_height + title_height

    plt.rcParams.update({'font.size': fontsize, 'font.family': 'Arial', 'pdf.fonttype': 42, #fonttype 42 keeps text editable
                         'mathtext.default': 'regular'}) #set $R^2$ in Arial instead of DejaVu
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Place images in the grid (top-left origin, like compose_final_figure)
    for index, img in enumerate(images):
        row, col = divmod(index, num_columns)
        x = sum(col_widths[:col]) * in_per_px
        y_top = title_height + sum(row_heights[:row]) * in_per_px
        w = img.shape[1] * in_per_px
        h = img.shape[0] * in_per_px
        ax = fig.add_axes([x / fig_width, 1 - (y_top + h) / fig_height, w / fig_width, h / fig_height])
        ax.imshow(img, interpolation='none')
        ax.axis('off')

    if colorbar_label != '':
        # vertically centered on the brain grid
        bar_height = grid_height * 0.8
        bar_bottom = (grid_height - bar_height) / 2
        cax = fig.add_axes([(grid_width + 0.15) / fig_width, bar_bottom / fig_height, 0.07 / fig_width, bar_height / fig_height])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(plain_superscripts(colorbar_label), fontsize=fontsize, labelpad=2)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.tick_params(labelsize=fontsize, length=2, width=0.5, pad=1)
        cbar.outline.set_linewidth(0.5)
        cbar.solids.set_rasterized(False) #matplotlib rasterizes colorbars by default; keep it vector
        cbar.solids.set_edgecolor('face') #hide hairline seams between colorbar segments

    if title:
        fig.text(grid_width / 2 / fig_width, 1 - (title_height / 2) / fig_height, title, ha='center', va='center', fontweight=title_weight, fontsize=fontsize)

    fig.savefig(output_filename, dpi=dpi, transparent=True)
    if png_dpi is not None:
        fig.savefig(os.path.splitext(output_filename)[0]+'.png', dpi=png_dpi, transparent=True)
    plt.close(fig)
def collect_source_data(source_csvs, sourcedata_dir, figure_name):
    """Copy the source data csv of each panel of a figure into sourcedata_dir as {figure_name}.csv,
    or {figure_name}A.csv, {figure_name}B.csv, ... when the figure has more than one panel."""
    import shutil
    os.makedirs(sourcedata_dir, exist_ok=True)
    for ind, source_csv in enumerate(source_csvs):
        panel = chr(ord('A') + ind) if len(source_csvs) > 1 else ''
        output_csv = os.path.join(sourcedata_dir, f'{figure_name}{panel}.csv')
        shutil.copy(source_csv, output_csv)
        print('source data saved:', output_csv)

def compose_final_figure_pdf(output_filename, input_pdfs, colorbar_pdf='', title=None, num_columns=2, row_padding_proportion=0, label_subplots=False, start_labeling_at=0, skip_first_label=False, max_width_mm=180, label_fontsize=7, title_fontsize=6, panel_top_padding_mm=None, row_titles=None, column_titles=None):
    #PDF version of compose_final_figure: takes the first page of each input pdf and places it on a grid
    #without rasterizing, so any editable text in the panels stays editable. Panel labels and the title
    #are added as editable text at label_fontsize/title_fontsize. If the grid is wider than max_width_mm,
    #the panels (and their text) are scaled down to fit and a warning is printed.
    #panel_top_padding_mm: list with one entry per panel of white space (mm) to add above that panel,
    #moving it down within its grid cell while its panel label stays at the top of the cell.
    #row_titles: one title per row, centered above that row; column_titles: one per column, centered below the grid
    #(both editable text at title_fontsize, in space added at full size so they are never scaled down).
    import io
    import math
    import matplotlib.pyplot as plt
    from pypdf import PdfReader, PdfWriter, PageObject, Transformation

    def first_page(path):
        page = PdfReader(path).pages[0]
        box = page.mediabox
        return page, float(box.left), float(box.bottom), float(box.width), float(box.height)

    panels = [first_page(path) for path in input_pdfs]
    top_pads = [pad / 25.4 * 72 for pad in (panel_top_padding_mm or [0] * len(panels))] # mm -> points
    num_rows = math.ceil(len(panels) / num_columns)

    # Compute row heights and column widths (points)
    row_heights = [0] * num_rows
    col_widths = [0] * num_columns
    for index, (_, _, _, w, h) in enumerate(panels):
        row, col = divmod(index, num_columns)
        row_heights[row] = max(row_heights[row], h + top_pads[index])
        col_widths[col] = max(col_widths[col], w)
    max_height = max(h for (_, _, _, _, h) in panels)
    row_padding = row_padding_proportion * max_height

    grid_width = sum(col_widths)
    grid_height = sum(row_heights) + num_rows * row_padding
    colorbar = first_page(colorbar_pdf) if colorbar_pdf != '' else None
    colorbar_scale = min(1, grid_height / colorbar[4]) if colorbar else 0 # shrink colorbar to the grid height if it's too tall
    native_width = grid_width + (colorbar[3] * colorbar_scale if colorbar else 0)

    # Scale everything down if it's wider than the allowed width (all sizes below are in final points)
    max_width = max_width_mm / 25.4 * 72
    scale = min(1, max_width / native_width)
    if scale > 0.999: # ignore rounding (e.g. 72 + 108 mm panels coming out a hair over 180 mm)
        scale = 1
    if scale < 1:
        print(f'warning: panels scaled by {scale:.2f} to fit {max_width_mm} mm; text inside the panels is shrunk by the same factor (e.g. 6 pt -> {6 * scale:.1f} pt)')
    title_height = 2 * title_fontsize if title else 0
    row_title_height = 2 * title_fontsize if row_titles else 0
    column_title_height = 2.5 * title_fontsize if column_titles else 0
    total_width = native_width * scale
    total_height = grid_height * scale + title_height + num_rows * row_title_height + column_title_height

    # Place panels (top-left of each grid cell, like compose_final_figure; pdf origin is bottom-left)
    final_page = PageObject.create_blank_page(width=total_width, height=total_height)
    label_positions = []
    row_title_positions = []
    y_offset = 0
    for row in range(num_rows):
        row_title_positions.append(title_height + y_offset * scale + row * row_title_height + row_title_height / 2)
        for col in range(num_columns):
            index = row * num_columns + col
            if index >= len(panels):
                continue
            page, left, bottom, w, h = panels[index]
            x = sum(col_widths[:col]) * scale
            y_top = title_height + y_offset * scale + (row + 1) * row_title_height
            final_page.merge_transformed_page(page, Transformation().translate(-left, -bottom).scale(scale).translate(x, total_height - y_top - (top_pads[index] + h) * scale))
            label_positions.append((index, x, y_top))
        y_offset += row_heights[row] + row_padding

    if colorbar:
        page, left, bottom, w, h = colorbar
        cb_scale = colorbar_scale * scale
        y = title_height + (grid_height * scale - h * cb_scale) / 2 # center vertically on the grid
        final_page.merge_transformed_page(page, Transformation().translate(-left, -bottom).scale(cb_scale).translate(grid_width * scale, total_height - y - h * cb_scale))

    # Overlay panel labels and title as editable text
    if label_subplots or title or row_titles or column_titles:
        plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42}) #fonttype 42 keeps text editable
        fig = plt.figure(figsize=(total_width / 72, total_height / 72))
        if label_subplots:
            for index, x, y_top in label_positions:
                if (index == 0) & skip_first_label:
                    continue
                label = chr(65 + index + start_labeling_at) + '.'  # 65 is ASCII 'A'
                fig.text((x + 1) / total_width, 1 - (y_top + 1) / total_height, label, # 1 pt inset so bold glyphs aren't clipped at the page edge
                         ha='left', va='top', fontweight='normal', fontsize=label_fontsize)
        if title:
            fig.text(total_width / 2 / total_width, 1 - (title_height / 4) / total_height, title,
                     ha='center', va='top', fontweight='normal', fontsize=title_fontsize)
        for row_title, y in zip(row_titles or [], row_title_positions):
            fig.text(0.5 * grid_width * scale / total_width, 1 - y / total_height, row_title,
                     ha='center', va='center', fontweight='normal', fontsize=title_fontsize)
        for col, column_title in enumerate(column_titles or []):
            x = (sum(col_widths[:col]) + col_widths[col] / 2) * scale
            fig.text(x / total_width, column_title_height / 2 / total_height, column_title,
                     ha='center', va='center', fontweight='normal', fontsize=title_fontsize)
        overlay = io.BytesIO()
        fig.savefig(overlay, format='pdf', transparent=True)
        plt.close(fig)
        final_page.merge_page(PdfReader(overlay).pages[0])

    writer = PdfWriter()
    writer.add_page(final_page)
    with open(output_filename, 'wb') as f:
        writer.write(f)
def compress_pdf_images(pdf_path, quality=90, downsample=0.72):
    #shrink a composed brain-map pdf in place: each raster image is flattened onto white, downsampled
    #(brain renders are ~625 dpi at print size, so 0.72 gives ~450 dpi) and stored as JPEG; text and vectors are untouched
    from pypdf import PdfWriter
    from PIL import Image
    writer = PdfWriter(clone_from=pdf_path)
    for page in writer.pages:
        for image_file in page.images:
            image_obj = image_file.indirect_reference.get_object()
            image = image_file.image.convert('RGBA')
            background = Image.new('RGBA', image.size, 'white')
            background.alpha_composite(image)
            image = background.convert('RGB')
            if downsample != 1:
                image = image.resize((round(image.width * downsample), round(image.height * downsample)), Image.LANCZOS)
            image_file.replace(image, quality=quality)
            if '/SMask' in image_obj:
                del image_obj['/SMask']
    writer.compress_identical_objects(remove_identicals=True, remove_orphans=True)
    with open(pdf_path, 'wb') as f:
        writer.write(f)
def crop_image(input_path, output_path, left, top, right, bottom):
    from PIL import Image
    with Image.open(input_path) as img:
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_path)
def compose_final_figure(output_filename, cropped_images, colorbar_image, title=None, num_columns=2,row_padding_proportion=0,color_dict=None,label_subplots=False,start_labeling_at=0,skip_first_label=False):
    from PIL import Image, ImageDraw, ImageFont
    import math
    # print(cropped_images)
    images = [Image.open(img_path) for img_path in cropped_images]
    widths, heights = zip(*(img.size for img in images))
    # Calculate the grid size
    num_images = len(images)
    num_rows = math.ceil(num_images / num_columns)
    
    # Get widths and heights per image
    image_sizes = [img.size for img in images]
    image_widths = [w for (w, h) in image_sizes]
    image_heights = [h for (w, h) in image_sizes]

    # Compute row heights and column widths
    row_heights = []
    col_widths = [0] * num_columns
    for row in range(num_rows):
        row_imgs = image_sizes[row * num_columns : (row + 1) * num_columns]
        row_heights.append(max(h for (w, h) in row_imgs))
        for col, (w, h) in enumerate(row_imgs):
            col_widths[col] = max(col_widths[col], w)

    total_width = sum(col_widths)
    if colorbar_image != '':
        colorbar = Image.open(colorbar_image)
        total_width += colorbar.size[0]

    legend_height = 200 if color_dict else 0
    # Calculate title height (if title is set)
    title_height = 0
    if title:
        try:
            font = ImageFont.truetype("Arial.ttf", 60)
        except IOError:
            font = ImageFont.load_default()
        bbox = font.getbbox(title)
        title_height = bbox[3] - bbox[1] + 30  # add padding

    total_height = int(sum(row_heights) + num_rows * row_padding_proportion * max(row_heights) + legend_height + title_height)

    max_width = max(widths)
    max_height = max(heights)

    

    # if colorbar_image != '':
    #     colorbar = Image.open(colorbar_image)
    #     total_width = num_columns * max_width + colorbar.size[0]
    # else:
    #     total_width = num_columns * max_width
    
      # Calculate cumulative heights
    cumulative_heights = []
    for row in range(num_rows):
        row_heights = heights[row * num_columns:(row + 1) * num_columns]
        cumulative_heights.append( max(row_heights))
    # legend_height = 200 if color_dict else 0
    # total_height = int(sum(cumulative_heights) + (num_rows)*row_padding_proportion*max_height + legend_height) 

    final_image = Image.new('RGBA', (total_width, total_height), (255, 255, 255, 0))

    # Place images in the grid
    y_offset = 0
    for row in range(num_rows):
        for col in range(num_columns):
            index = row * num_columns + col
            if index < num_images:
                img = images[index]
                x_offset = sum(col_widths[:col])
                position = (x_offset, int(y_offset + title_height))                # position = (col * max_width, int(y_offset))
                final_image.paste(img, position)
                if label_subplots:
                    if((index==0)&(skip_first_label)):
                        continue
                    else:
                        label = chr(65 + index+start_labeling_at)+'.'  # 65 is ASCII 'A'
                        draw = ImageDraw.Draw(final_image)
                        try:
                            font = ImageFont.truetype("Arial.ttf", 90)
                        except IOError:
                            font = ImageFont.load_default()

                        draw.text((position[0] + 15, position[1] + 10), label, fill="black", font=font)
        y_offset += cumulative_heights[row] + row_padding_proportion*max_height
    
    if colorbar_image != '':
        colorbar = Image.open(colorbar_image)

        # Compute available vertical space (grid height only)
        grid_height = total_height - legend_height - title_height

        # Resize colorbar to match grid height if it's too tall
        if colorbar.size[1] > grid_height:
            colorbar_ratio = colorbar.size[0] / colorbar.size[1]
            new_height = grid_height
            new_width = int(new_height * colorbar_ratio)
            colorbar = colorbar.resize((new_width, new_height), resample=Image.BICUBIC)

        # Center vertically
        colorbar_y = (grid_height - colorbar.size[1]) // 2
        colorbar_x = sum(col_widths)
        final_image.paste(colorbar, (colorbar_x, colorbar_y))
    if title:
        draw = ImageDraw.Draw(final_image)
        try:
            font = ImageFont.truetype("Arial.ttf", 60)
        except IOError:
            font = ImageFont.load_default()

        # Measure text width
        bbox = font.getbbox(title)  # (left, top, right, bottom)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center the title
        grid_width = sum(col_widths)
        x = (grid_width - text_width) // 2
        y = 15  # a bit of padding from the top edge
        draw.text((x, y), title, fill="black", font=font)
    
    # Draw horizontal legend at bottom
    if color_dict:
        legend_y = total_height - legend_height + 80
        draw = ImageDraw.Draw(final_image)
        try:
            font = ImageFont.truetype("Arial.ttf", 90)
        except IOError:
            font = ImageFont.load_default()

        # Measure total width needed
        box_width = 150
        box_spacing = 50
        total_legend_width = 0
        label_widths = []
        for label in color_dict.keys():
            bbox = font.getbbox(label)
            text_width = bbox[2] - bbox[0]
            label_widths.append(text_width)
            total_legend_width += box_width + 10 + text_width + box_spacing
        total_legend_width -= box_spacing  # remove extra spacing after last item

        x_offset = (total_width - total_legend_width) // 2

        for label, text_width in zip(color_dict.keys(), label_widths):
            color = color_dict[label]
            if isinstance(color, tuple) and all(isinstance(c, float) for c in color):
                color = tuple(int(c * 255) for c in color)

            draw.rectangle([x_offset, legend_y, x_offset + box_width, legend_y + box_width/2], fill=color, outline='black', width=8)
            bbox = font.getbbox(label)
            text_height = bbox[3] - bbox[1]

            text_x = x_offset + box_width + 30
            text_y = legend_y + (30 - text_height) // 2  # 30 = height of the rectangle

            draw.text((text_x, text_y), label, fill="black", font=font)

            x_offset += box_width + 20 + text_width + box_spacing

    final_image.save(output_filename)
    return final_image
def plot_preference_surf(textures,filepath,ROI_niis=[],color_dict=None,cmap='cold_hot',views=['lateral','ventral'],threshold=None,vmin=None,vmax=None,title=None,ROIs=[],ROI_colors=[],roi_thickness=0,width_mm=None,fontsize=7):
    #width_mm: if set, also save filepath+'.pdf' at that printed width with the title as editable text at fontsize (pt)
    import nilearn.datasets
    from nilearn import surface
    from nilearn import plotting
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage') 
    temp_filename = filepath

    plt.rcParams.update({'font.size': 10,'font.family': fontfamily})
    # views = [('left','lateral'),('right','lateral'),('left','ventral'),('right','ventral')]
    for (hemi, view, texture) in [(hemi,view,texture) for view in views for (hemi,texture) in zip(['left','right'],textures)]:
    # for (hemi,view),texture in zip(views,[textures[0],textures[1],textures[0],textures[1]]):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(5, 5))

        transform_mesh = fsaverage['pial_'+hemi]
        plot_mesh = fsaverage['infl_'+hemi]
        bg_map = fsaverage['sulc_'+hemi]
        inner_mesh = fsaverage['white_' + hemi]

        # print(texture.shapes
        # print(plot_mesh.shape)

        # texture = np.nan_to_num(texture, nan=0)
        plotting.plot_surf_roi(plot_mesh, texture, hemi=hemi,
                          view=view, colorbar=False,
                          threshold=threshold,
                          label=title,
                          bg_map=bg_map,
                          cmap=cmap,
                          vmax=vmax,
                          axes=ax,
                          engine='matplotlib')
        roi_textures = []
        mesh_faces = surface.load_surf_mesh(plot_mesh)[1] if roi_thickness else None
        for ind,ROI_nii in enumerate(ROI_niis):
            n_points_to_sample = 50
            roi_texture = surface.vol_to_surf(ROI_nii,transform_mesh,inner_mesh=inner_mesh, depth = np.linspace(0, 1, n_points_to_sample),interpolation='linear')
            roi_texture = (roi_texture>0.333)*1 #binarize the surface ROI map, anything that was part of the ROI in volume should be in surface
            roi_textures.append(roi_texture)
            try:
                if roi_thickness:
                    # Paint a band of faces around the boundary — a contour line
                    # one face wide is invisible once the brain is projected.
                    face_idx = _thick_boundary_face_indices(roi_texture, mesh_faces, roi_thickness)
                    _paint_boundary_faces(ax, face_idx, ROI_colors[ind])
                else:
                    # plotting.plot_surf_contours(plot_mesh,roi_texture,avg_method='mean',axes=ax,levels=[ind for ind in np.arange(1,len(ROIs)+1)],labels=ROIs,colors=ROI_colors)
                    plotting.plot_surf_contours(plot_mesh,roi_texture,avg_method='mean',axes=ax,levels=[1],labels=[ROIs[ind]],colors=[ROI_colors[ind]])
            except Exception as e:
                print(e)
                pass
        if view=='ventral':
            if hemi == 'left':
                ax.view_init(elev=270,azim=180)# Rotate the (left, ventral) view by 180 degrees
            elif hemi == 'right':
                ax.view_init(elev=270,azim=0)
            ax.set_box_aspect((1, 1, 1), zoom=1.3) #zoom in because brains are too small

        plt.savefig(filepath+'_'+hemi+'_'+view+'.png', bbox_inches='tight', transparent=True, dpi=300)
        plt.close()

    ##crop each separate brain plot
    for (hemi,view) in [(hemi,view) for view in views for hemi in ['left','right']]:
        if(view=='lateral'):
            left=150
            top=240
            width=1100
            height=950
        elif(view=='ventral'):
            left=150
            top=400
            width=1100
            height=775
        elif(view=='medial'):
            left=150
            top=215
            width=1100
            height=950
        crop_image(temp_filename+'_'+hemi+'_'+view+'.png',temp_filename+'_'+hemi+'_'+view+'.png',left,top,width,height)
    
    ### put together all of the cropped images into one plot
    list_images = [temp_filename+'_'+hemi+'_'+view+'.png' for view in views for hemi in ['left','right']]
    colorbar_filepath = ''
    compose_final_figure(temp_filename+'.png', list_images, color_dict=color_dict,colorbar_image=colorbar_filepath, title=title)
    if width_mm is not None:
        compose_final_figure_editable(temp_filename+'.pdf', list_images, cmap=None, vmin=None, vmax=None, title=title, width_mm=width_mm, fontsize=fontsize)
    for (hemi,view) in [(hemi,view) for view in views for hemi in ['left','right']]:
            delete_file = temp_filename+'_'+hemi+'_'+view+'.png'
            os.remove(delete_file)

def _hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

def bivariate_blend(x, y, corner_neither=(0.05, 0.05, 0.05),
                    corner_x=None, corner_y=None, corner_both=None):
    """Bilinear blend of four corner colors over the unit square.

    x, y are arrays in [0, 1] (same shape). Returns (..., 3) RGB array.
    Corner layout: (x=0,y=0)=neither, (x=1,y=0)=x, (x=0,y=1)=y, (x=1,y=1)=both.
    """
    import numpy as np
    if corner_x is None:
        corner_x = _hex2rgb('#173C92')
    if corner_y is None:
        corner_y = _hex2rgb('#D65B06')
    if corner_both is None:
        corner_both = _hex2rgb('#2A8F3F')

    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    y = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)

    c00 = np.asarray(corner_neither, dtype=float)
    c10 = np.asarray(corner_x, dtype=float)
    c01 = np.asarray(corner_y, dtype=float)
    c11 = np.asarray(corner_both, dtype=float)

    xx = x[..., None]
    yy = y[..., None]
    rgb = ((1 - xx) * (1 - yy) * c00
           + xx * (1 - yy) * c10
           + (1 - xx) * yy * c01
           + xx * yy * c11)
    return rgb

def save_bivariate_legend(filename, label_x='Vision', label_y='Language',
                          vmax_x=None, vmax_y=None, resolution=256,
                          corner_neither=(0.05, 0.05, 0.05),
                          corner_x=None, corner_y=None, corner_both=None):
    """Render the 2D bivariate color square as a PNG with axis labels."""
    import numpy as np
    import matplotlib.pyplot as plt

    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, resolution),
                                 np.linspace(0, 1, resolution))
    rgb = bivariate_blend(grid_x, grid_y,
                          corner_neither=corner_neither,
                          corner_x=corner_x, corner_y=corner_y,
                          corner_both=corner_both)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(rgb, origin='lower', extent=[0, 1, 0, 1], aspect='equal',
              interpolation='nearest')
    ax.set_xlabel(label_x, fontsize=fontsize-2, family=fontfamily)
    ax.set_ylabel(label_y, fontsize=fontsize-2, family=fontfamily)

    if vmax_x is not None:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0', f'{vmax_x:.3f}'], fontsize=fontsize-6, family=fontfamily)
    else:
        ax.set_xticks([])
    if vmax_y is not None:
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', f'{vmax_y:.3f}'], fontsize=fontsize-6, family=fontfamily)
    else:
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=300)
    plt.close()

def plot_bivariate_surface(nii_y, nii_x, filename, vmax_y=None, vmax_x=None,
                           threshold=1e-6, label_y='Language', label_x='Vision',
                           title='', views=['lateral', 'ventral'],
                           corner_neither=(0.05, 0.05, 0.05),
                           corner_x=None, corner_y=None, corner_both=None):
    """Plot two volumes onto the fsaverage surface as per-vertex RGB.

    nii_y and nii_x are Nifti1Image objects (already on the same affine).
    Each axis is independently normalized to [0, vmax_*] then blended via
    bivariate_blend. Vertices below threshold on both axes become transparent.
    """
    import nilearn.datasets
    from nilearn import surface
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    import gc

    temp_filename = filename
    fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')

    n_points_to_sample = 50
    textures_y = {}
    textures_x = {}
    for hemi in ('left', 'right'):
        transform_mesh = fsaverage['pial_' + hemi]
        inner_mesh = fsaverage['white_' + hemi]
        ty = surface.vol_to_surf(nii_y, transform_mesh, inner_mesh=inner_mesh,
                                 depth=np.linspace(0, 1, n_points_to_sample),
                                 interpolation='nearest')
        tx = surface.vol_to_surf(nii_x, transform_mesh, inner_mesh=inner_mesh,
                                 depth=np.linspace(0, 1, n_points_to_sample),
                                 interpolation='nearest')
        textures_y[hemi] = np.nan_to_num(ty, nan=0.0)
        textures_x[hemi] = np.nan_to_num(tx, nan=0.0)

    if vmax_y is None:
        vmax_y = max(np.nanpercentile(textures_y[h], 99) for h in ('left', 'right'))
    if vmax_x is None:
        vmax_x = max(np.nanpercentile(textures_x[h], 99) for h in ('left', 'right'))
    vmax_y = max(vmax_y, 1e-12)
    vmax_x = max(vmax_x, 1e-12)

    plt.rcParams.update({'font.size': 10, 'font.family': fontfamily})

    for (hemi, view) in [(hemi, view) for view in views for hemi in ['left', 'right']]:
        coords, faces = surface.load_surf_mesh(fsaverage['infl_' + hemi])
        bg = surface.load_surf_data(fsaverage['sulc_' + hemi])

        ty = np.clip(textures_y[hemi], 0.0, None) / vmax_y
        tx = np.clip(textures_x[hemi], 0.0, None) / vmax_x

        rgb = bivariate_blend(tx, ty,
                              corner_neither=corner_neither,
                              corner_x=corner_x, corner_y=corner_y,
                              corner_both=corner_both)
        alpha = ((tx > (threshold / vmax_x)) | (ty > (threshold / vmax_y))).astype(float)
        vertex_rgba = np.concatenate([rgb, alpha[:, None]], axis=1)

        face_rgba = vertex_rgba[faces].mean(axis=1)
        # If any vertex of a face is transparent, fade the face proportionally
        face_alpha = vertex_rgba[faces, 3].mean(axis=1)
        face_rgba[:, 3] = face_alpha

        # Background grayscale shading from sulc (same recipe as nilearn bg_map)
        bg_norm = (bg - bg.min()) / (bg.max() - bg.min() + 1e-12)
        face_bg = bg_norm[faces].mean(axis=1)
        face_bg_rgba = np.stack([0.4 + 0.4 * (1 - face_bg)] * 3 + [np.ones_like(face_bg)], axis=1)

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(5, 5))
        tri_bg = Poly3DCollection(coords[faces], linewidths=0., antialiased=False)
        tri_bg.set_facecolors(face_bg_rgba)
        ax.add_collection3d(tri_bg)

        tri = Poly3DCollection(coords[faces], linewidths=0., antialiased=False)
        tri.set_facecolors(face_rgba)
        ax.add_collection3d(tri)

        # Frame the brain like nilearn does
        xlim = (coords[:, 0].min(), coords[:, 0].max())
        ylim = (coords[:, 1].min(), coords[:, 1].max())
        zlim = (coords[:, 2].min(), coords[:, 2].max())
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))
        ax.set_axis_off()

        # View orientation matches nilearn conventions
        if view == 'lateral':
            ax.view_init(elev=0, azim=180 if hemi == 'left' else 0)
        elif view == 'medial':
            ax.view_init(elev=0, azim=0 if hemi == 'left' else 180)
        elif view == 'ventral':
            if hemi == 'left':
                ax.view_init(elev=270, azim=180)
            else:
                ax.view_init(elev=270, azim=0)
            ax.set_box_aspect((1, 1, 1), zoom=1.3)
        elif view == 'dorsal':
            ax.view_init(elev=90, azim=0)

        plt.savefig(temp_filename + '_' + hemi + '_' + view + '.png',
                    bbox_inches='tight', transparent=True, dpi=300)
        plt.close()
        gc.collect()

    # Crop each separate brain plot (same boxes as plot_surface)
    for (hemi, view) in [(hemi, view) for view in views for hemi in ['left', 'right']]:
        if view == 'lateral':
            left, top, width, height = 150, 240, 1100, 950
        elif view == 'ventral':
            left, top, width, height = 150, 400, 1100, 775
        elif view == 'medial':
            left, top, width, height = 150, 215, 1100, 950
        else:
            left, top, width, height = 150, 240, 1100, 950
        crop_image(temp_filename + '_' + hemi + '_' + view + '.png',
                   temp_filename + '_' + hemi + '_' + view + '.png',
                   left, top, width, height)

    # 2D legend
    legend_path = temp_filename + '_legend.png'
    save_bivariate_legend(legend_path, label_x=label_x, label_y=label_y,
                          vmax_x=vmax_x, vmax_y=vmax_y,
                          corner_neither=corner_neither,
                          corner_x=corner_x, corner_y=corner_y,
                          corner_both=corner_both)

    list_images = [temp_filename + '_' + hemi + '_' + view + '.png'
                   for view in views for hemi in ['left', 'right']]
    compose_final_figure(filename + '.png', list_images, legend_path, title=title)

    for path in list_images:
        os.remove(path)
    os.remove(legend_path)

def plot_preference_img_volume(img,filepath,color_dict,labels,threshold=None,vmin=None,vmax=None,cmap='cold_hot',title=None,):
    import nibabel
    import numpy as np
    from nilearn import plotting
    import matplotlib.pyplot as plt

    #NOTE: view_surf simply plots a surface mesh on a brain, no tri averaging like plot_surf_stat_map
    #NOTE: you can't have any NaN values when using view_surf -- it doesn't handle the threshold correctly
    # so, converting all NaN's negative infinity and then turning all negatives to 0 for plotting

    fig = plt.figure(1,figsize=(6.4,3.7))
    display = plotting.plot_glass_brain(
            stat_map_img = img,
            # output_file=filepath,
            colorbar=False,
            cmap=cmap,
            threshold=threshold,
            display_mode='lyrz',#'lyrz',lzr
            vmin=vmin,
            vmax=vmax,
            title=title,
            figure=fig
            # norm=norm
            )


    scale_factor = 7.5
    ax = display.axes['l'].ax
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    x_scale = abs(xlim[1]-xlim[0])/scale_factor
    y_scale = abs(ylim[1]-ylim[0])/scale_factor
    xlim_start = xlim[0]-x_scale
    ylim_start = ylim[0]-y_scale
    for ind,label in enumerate(labels[1:6]):
        ax.text(xlim_start, ylim_start-(y_scale*ind), label, fontsize='small',
                bbox={'fc':color_dict[label], 'pad':2})

    ax = display.axes['y'].ax
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    x_scale = abs(xlim[1]-xlim[0])/scale_factor
    y_scale = abs(ylim[1]-ylim[0])/scale_factor
    xlim_start = xlim[0] + x_scale
    ylim_start = ylim[0]-y_scale
    for ind,label in enumerate(labels[6:11]):
        ax.text(xlim_start, ylim_start-(y_scale*ind), label,fontsize='small',
                bbox={'fc':color_dict[label], 'pad':2})

    ax = display.axes['r'].ax
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    x_scale = abs(xlim[1]-xlim[0])/scale_factor
    y_scale = abs(ylim[1]-ylim[0])/scale_factor
    xlim_start = xlim[0] + x_scale
    ylim_start = ylim[0]-y_scale
    for ind,label in enumerate(labels[11:16]):
        ax.text(xlim_start, ylim_start-(y_scale*ind), label,fontsize='small',
                bbox={'fc':color_dict[label], 'pad':2})

    ax = display.axes['z'].ax
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    x_scale = abs(xlim[1]-xlim[0])/scale_factor
    y_scale = abs(ylim[1]-ylim[0])/scale_factor
    xlim_start = xlim[0] + x_scale
    ylim_start = ylim[0]-y_scale
    for ind,label in enumerate(labels[16:20]):
        ax.text(xlim_start, ylim_start-(y_scale*ind), label,fontsize='small',
                bbox={'fc':color_dict[label], 'pad':2})

    display.savefig(filepath)
    plt.close()
@contextlib.contextmanager
def _finite_native_width():
    """Keep seaborn's strip jitter finite when every facet holds one category.

    seaborn 0.12 measures the categorical unit width from the spacing between
    converted x positions. With ``sharex=False`` each facet gets its own unit
    converter, so when every panel holds a single category (which is what
    averaging the hemispheres together leaves us with) they all convert to
    position 0, the pooled spacing is undefined, and ``_native_width`` comes
    back as nan. stripplot then asks numpy for a uniform draw over nan bounds
    and raises ``OverflowError: Range exceeds valid bounds``. Fall back to
    seaborn's own single-category width of 1; a no-op whenever the measured
    width is already finite.
    """
    import numpy as np
    try:
        from seaborn.categorical import _CategoricalPlotterNew as plotter
        original = plotter._native_width
    except (ImportError, AttributeError):
        yield  # different seaborn version; nothing to guard
        return

    plotter._native_width = property(
        lambda self: original.fget(self) if np.isfinite(original.fget(self)) else 1.0
    )
    try:
        yield
    finally:
        plotter._native_width = original


def plot_bar_and_strip(
    data,
    column_group,
    col_order,
    width_ratios,
    params,
    col_wrap = None,
    kind_order=('bar', 'strip'),
    height=4,
    aspect=1,
    colorbar_width=0.02,
    colorbar_padding=0.02,
    sharey=True,
    legend_below=False,
    also_plot_median=False,
    overlay_points=False
):
    """overlay_points: also draw the individual data points (one per participant and
    hemisphere) on top of the bars of the 'bar' plot, in the same dodged slots."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import matplotlib as mpl
    import pandas as pd
    import re
    _update_font_rc((fontsize-4)*font_scale)

    saturation =0.75
    hue_column = params.get('hue', None)
    hue_order = params.get('hue_order', None)
    palette = params.get('palette', None)

    apply_colorbar = False

    if hue_column is None or hue_order is None or palette is None:
        raise ValueError("params must include 'hue', 'hue_order', and 'palette'")

    hue_order_used = hue_order

    parsed_results = []
    parse_success = True
    for val in hue_order_used:
        match = re.match(r"(.+?) layer (\d+)", val)
        if match:
            model = match.group(1)
            layer = int(match.group(2))
            parsed_results.append((val, model, layer))
        else:
            parse_success = False
            break

    if parse_success:
        apply_colorbar = True
        parsed_df = (
            pd.DataFrame(parsed_results, columns=['feature', 'model', 'layer'])
            .sort_values(by=['model','layer'])
        )

    figs = {}

    if also_plot_median and 'median_bar' not in kind_order:
        kind_order = tuple(kind_order) + ('median_bar',)

    for kind in kind_order:
        catplot_kind = 'bar' if kind == 'median_bar' else kind
        plot_kwargs = dict(
            kind=catplot_kind,
            data=data,
            col=column_group,
            col_order=col_order,
            col_wrap=col_wrap,
            height=height,
            aspect=aspect,
            edgecolor="black",
            linewidth=_lw(2.5),
            sharex=False,
            sharey=sharey,
            facet_kws={'gridspec_kws': {'width_ratios': width_ratios}},
            **params
        )

        if kind == 'bar':
            plot_kwargs.update(dict(errorbar='se', errcolor="black",saturation=saturation,width=bar_width))
        if kind == 'median_bar':
            plot_kwargs.update(dict(errorbar=('pi', 50), errcolor="black", saturation=saturation, estimator='median',width=bar_width))
        if kind == 'strip':
            plot_kwargs.update(dict(dodge=True))

        with _finite_native_width():
            g = sns.catplot(**plot_kwargs)

        if kind == 'bar' and overlay_points:
            # map_dataframe hands stripplot the same per-facet data (and so the same
            # x order and hue slots) that catplot gave the bars
            with _finite_native_width():
                g.map_dataframe(sns.stripplot, x=params['x'], y=params['y'], hue=hue_column,
                                hue_order=hue_order, palette={h: 'black' for h in hue_order},
                                dodge=True, size=1.2 if figure_fontsize is not None else 4,
                                alpha=0.5, linewidth=0, legend=False, zorder=4)

        if figure_width_mm is not None and not (apply_colorbar | legend_below) and g._legend is not None:
            # the margin FacetGrid reserves for its legend no longer fits once save_figure
            # resizes the figure, so put the legend just outside the axes instead
            sns.move_legend(g, 'center left', bbox_to_anchor=(1, 0.5), frameon=False)
            g.figure.subplots_adjust(right=0.98)

        if apply_colorbar | legend_below:
            g._legend.remove()
    
        figs[kind] = g
        plt.subplots_adjust(wspace=0.2)

    if apply_colorbar:
        fig = figs[kind_order[0]].fig

        models = parsed_df['model'].unique()
        n_models = len(models)

        # Compute total colorbar width
        total_cbar_width = n_models * (colorbar_width + colorbar_padding)

        # Shrink main plot to leave space on right
        right_limit = 0.92 - total_cbar_width
        fig.subplots_adjust(right=right_limit)

        # Now place colorbars outside main plot
        start_x = right_limit + colorbar_padding

        for i, model in enumerate(models):
            model_df = parsed_df[parsed_df['model'] == model]
            feature_colors = [
                sns.desaturate(palette[feat], saturation) for feat in model_df['feature']
            ]

            layers = model_df['layer'].values
            layers_sorted = np.sort(layers)

            cmap = mpl.colors.ListedColormap(feature_colors)
            boundaries = np.append(layers_sorted - 0.5, layers_sorted[-1] + 0.5)
            norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)

            scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

            left = start_x + i * (colorbar_width + colorbar_padding)
            cbar_ax = fig.add_axes([left, 0.15, colorbar_width, 0.7])
            cbar = plt.colorbar(scalar_map, cax=cbar_ax, orientation='vertical', ticks=layers_sorted)
            cbar.set_label(f"{model} layers")
            cbar.ax.yaxis.set_label_position('left')
            cbar.outline.set_linewidth(2)
            cbar.ax.invert_yaxis()
    if legend_below:
        axes = figs.get('bar').axes[0]  # first row of facets
        if len(axes) >= 2:
            import matplotlib.patches as mpatches

            # Make custom handles
            pointlight_handles = [
                mpatches.Patch(facecolor=palette['interacting pointlights'], label='interacting pointlights', edgecolor='k',linewidth=_lw(2.5)),
                mpatches.Patch(facecolor=palette['non-interacting pointlights'], label='non-interacting pointlights', edgecolor='k',linewidth=_lw(2.5))
            ]
            speech_handles = [
                mpatches.Patch(facecolor=palette['intact speech'], label='intact speech',edgecolor='k',linewidth=_lw(2.5)),
                mpatches.Patch(facecolor=palette['degraded speech'], label='degraded speech',edgecolor='k',linewidth=_lw(2.5))
            ]

            # These two legends sit centred under the first two panels, and the
            # 'non-interacting pointlights' label is wider than its panel. Push
            # them apart as the font grows so they cannot run into each other;
            # zero offset, i.e. the published layout, at font_scale == 1.
            legend_spread = 0.25 * (font_scale - 1)

            # Place left legend under the first axis
            axes[0].legend(handles=pointlight_handles,
                        loc='upper center',
                        bbox_to_anchor=(0.5 - legend_spread, -0.25),
                        frameon=False,
                        ncol=1,
                        fontsize=_fs(fontsize*font_scale))

            # Place right legend under the second axis
            axes[1].legend(handles=speech_handles,
                        loc='upper center',
                        bbox_to_anchor=(0.5 + legend_spread, -0.25),
                        frameon=False,
                        ncol=1,
                        fontsize=_fs(fontsize*font_scale))


    if also_plot_median:
        return figs.get('bar'), figs.get('strip'), figs.get('median_bar')
    return figs.get('bar'), figs.get('strip')
def plot_stacked_bars(
    self,
    results,
    localizers,
    localizer_masks,
    order_dict,
    palette,
    feature_names,
    hue,
    width_ratios,
    stats_to_do=None,
    pvalue_dict=None,
    plot_noise_ceiling=False,
    noise_ceiling_means=None,
    noise_ceiling_sems=None,
    restrict_legend=False,
    response_label='ind_product_measure',
    file_label='',
    file_suffix='',
    use_median=False,
    label_region=True,
    reveal_localizers=None,
):
    """``reveal_localizers`` draws bars for only the first N localizer panels and
    hides the rest, for talk build-ups. Every panel is still laid out and drawn
    the same way, so the axes, ticks, y limits and legend are pixel-identical
    across frames and only the bars appear. None (the default) reveals all."""
    from matplotlib import gridspec
    from matplotlib.patches import Patch
    
    _update_font_rc(12*font_scale)

    features = [self.labels_dict.get(f, f) for f in feature_names]
    # features.reverse()  # stack bottom-up in desired order

    n_locs = len(localizers)
    fig_height = stacked_fig_height
    fig_width = n_locs * 3
    fig = plt.figure(figsize=(fig_width,fig_height))
    gs = gridspec.GridSpec(1, n_locs, width_ratios=width_ratios)
    # axes = [fig.add_subplot(gs[0, i]) for i in range(n_locs)]
    
    axes = [fig.add_subplot(gs[0])]
    for ind, axis in enumerate(gs):
        if(ind > 0):
            axes.append(fig.add_subplot(axis, sharey=axes[0]))
    
    pivoted = pd.pivot_table(
        data=results,
        values='encoding_response',
        columns=[hue, 'hemi_mask'],
        aggfunc='mean'
    )
    # Step 2: Stack both levels of column multi-index into rows
    averaged_results = pivoted.stack(level=[0, 1]).reset_index()

    # Step 3: Rename for clarity
    averaged_results.columns = ['subject', hue, 'hemi_mask', 'value']

    if use_median:
        # Collapse across subjects with median: one row per (hue, hemi_mask)
        averaged_results = (
            averaged_results
            .groupby([hue, 'hemi_mask'], as_index=False)['value']
            .median()
        )

    for idx, (localizer, ax) in enumerate(zip(localizers, axes)):
        order = order_dict[localizer]
        temp_data = averaged_results.copy()
        temp_data['hemi_mask'] = pd.Categorical(
            temp_data['hemi_mask'], categories=order
        )
        # Filter positive contributions only
        pos_features = []
        for mask in order:
            for feat in features:
                temp = results.loc[
                    (results['hemi_mask'] == mask) &
                    (results[hue] == feat), 'encoding_response'
                ]
                agg_val = temp.median() if use_median else temp.mean()
                if agg_val > 0:
                    pos_features.append(feat)

        plot_feats = [f for f in features if f in pos_features]
        plot_feats.reverse()  # back to normal for plotting
        sns.histplot(
            data=temp_data,
            x='hemi_mask',
            hue=hue,
            hue_order=plot_feats,
            palette=palette,
            multiple='stack',
            weights='value',
            shrink=bar_width,
            ax=ax,
            stat='count',
            linewidth=_lw(1.75),
            edgecolor='black',
            alpha=1
        )
        # Remove grid for clean appearance
        ax.grid(False)
        # Significance shading (optional)
        if stats_to_do and pvalue_dict:
            for patch, (x_tick, f_label) in zip(ax.patches, [(val, f) for f in plot_feats[::-1] for val in order]):
                pval = pvalue_dict.get(((x_tick, f_label),(x_tick, f_label)))
                if pval >= 0.05:
                    patch.set_facecolor('gray')

        # Noise ceiling
        if plot_noise_ceiling and noise_ceiling_means and noise_ceiling_sems:
            for mean, sem, patch in zip(noise_ceiling_means[idx], noise_ceiling_sems[idx], ax.patches[::len(plot_feats)]):
                bar_center = patch.get_x() + patch.get_width() / 2
                bar_half = patch.get_width() * 0.6
                ax.plot([bar_center - bar_half, bar_center + bar_half], [mean, mean], color='white', linestyle='dotted')
                ax.fill_between(
                    [bar_center - bar_half, bar_center + bar_half],
                    mean - sem, mean + sem, color='black', alpha=0.3
                )

        # Legend
        if restrict_legend and pvalue_dict:
            shown_feats = set()
            for ((mask, feat),(mask, feat)), pval in pvalue_dict.items():
                if pval < 0.05:
                    shown_feats.add(feat)
            final_legend_feats = [f for f in [self.labels_dict.get(item,item) for item in self.plot_features] if f in shown_feats]
        else:
            final_legend_feats = self.plot_features
            
        if idx == 0:
            ax.set_ylabel('cumulative explained variance $R^2$')
        else:
            ax.set_ylabel('')
            ax.yaxis.set_visible(False)  # Only show ticks on first axis
            ax.spines['left'].set_visible(False)

        # Set consistent xticks
        # new_xticks = [label.split(' ')[-1].split('-')[0] for label in order]
        # ax.set_xticks(ax.get_xticks())
        # ax.set_xticklabels(new_xticks)

        # Hemisphere labels
        # for text_label, x_position in zip(['left', 'right'], [0.25, 0.75]):
        #     ax.text(x_position, -0.05, text_label, transform=ax.transAxes, ha='center', va='top')

        # Set legend only once (on final axis)
        if idx == len(localizers) - 1:
            legend_elements = [
                Patch(
                    facecolor=palette[self.labels_dict.get(f, f)],
                    edgecolor='k',
                    label=self.labels_dict.get(f, f).replace('_', ' ')
                )
                for f in final_legend_feats
            ]
            if nonsignificant_legend_label and stats_to_do and pvalue_dict:
                legend_elements.append(Patch(
                    facecolor='gray',
                    edgecolor='k',
                    label=nonsignificant_legend_label,
                ))
            ax.legend(
                handles=legend_elements,
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                frameon=False
            )
        else:
            ax.legend_.remove()  # Remove legend for all other subplots
        # Cosmetic formatting
        new_xticks = [region_tick_label(label) for label in order]
        # ax.set_xticklabels([f"{t.get_text().split(' ')[1]}\n{t.get_text().split(' ')[0]}" for t in ax.get_xticklabels()])

        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(new_xticks)
        ax.set_xlabel('')
        ax.set_ylabel('cumulative explained variance $R^2$' if idx == 0 else '')
        ax.set_ylim((0,0.18))
        if fix_category_xlim:
            ax.set_xlim(-0.5, len(order) - 0.5)
        ax.spines[['top', 'right']].set_visible(False)

        if label_region:
            region_label_dict = {
                'motion': 'motion\nresponsive',
                'social interaction': 'social\ninteraction\nselective',
                'language': 'language\nselective',
            }
            contrast_label_dict = {
                'motion': '(interacting +\nnot interacting)',
                'social interaction': '(interacting -\nnot interacting)',
                'language': '(intact - degraded)',
            }
            region_text = region_label_dict.get(localizer, '\n'.join(localizer.split()) + '\nregions')
            contrast_text = contrast_label_dict.get(localizer, '')
            ax.text(0.5, 0.95, region_text, fontsize=_fs(11*font_scale), fontweight='normal', color='black',
                    ha='center', va='bottom', transform=ax.transAxes)
            ax.text(0.5, 0.93, contrast_text, fontsize=_fs(11*font_scale), fontweight='normal', color='black',
                    ha='center', va='top', transform=ax.transAxes)

        if reveal_localizers is not None and idx >= reveal_localizers:
            for patch in ax.patches:
                patch.set_visible(False)

    # Save
    median_tag = '_median' if use_median else ''
    filename = f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_{file_suffix}{median_tag}_stacked.pdf"
    filepath = os.path.join(self.figure_dir, filename)
    save_figure(fig, filepath)
    plt.close()
def plot_voxelwise_feature_scatter(pkl_path, region_names, feature_names, feature_sources,
                                    save_dir=None, plot_scatter=False, split_hemisphere=False,
                                    axis_labels=None, file_tag='', save_source_data=False):
        """axis_labels: (x label, y label) for the two features; file_tag is prepended to the saved filename.
        When figure_width_mm is set, the figure is saved as an editable pdf at that width.
        save_source_data: also save the plotted voxels (one row per voxel) as a csv next to the figure."""
        import pickle
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        import os
        from scipy.stats import pearsonr
        from itertools import combinations

        # Load voxelwise data
        with open(pkl_path, 'rb') as f:
            voxelwise_list = pickle.load(f)

        all_records = []
        seen = set()

        for entry in voxelwise_list:
            region = entry['mask']
            if region not in region_names:
                continue
            # one voxel set per participant and region: the all-runs localizer (not a cross-validated split),
            # so voxels are never averaged by position across different selections
            if len(str(entry.get('averaged_localizer_runs', '')).split(',')) > 1:
                continue
            key = (entry['subject'], entry['hemisphere'], region, entry.get('enc_feature_name'), entry.get('glm_task_response'), entry.get('glm_response_contrast'))
            if key in seen:
                continue
            seen.add(key)

            subject = entry['subject']
            hemi = entry['hemisphere']
            feature_enc = entry.get('enc_feature_name')
            feature_glm = entry.get('glm_response_contrast')
            voxels_enc = entry.get('enc_voxelwise')
            voxels_glm = entry.get('glm_voxelwise')

            if feature_enc and voxels_enc is not None:
                all_records.append({
                    'subject': subject,
                    'hemisphere': hemi,
                    'region': region,
                    'feature': feature_enc,
                    'source': 'enc_voxelwise',
                    'voxels': voxels_enc,
                    'voxel_idx': np.arange(len(voxels_enc))
                })
            if feature_glm and voxels_glm is not None:
                all_records.append({
                    'subject': subject,
                    'hemisphere': hemi,
                    'region': region,
                    'feature': feature_glm,
                    'source': 'glm_voxelwise',
                    'voxels': voxels_glm,
                    'voxel_idx': np.arange(len(voxels_glm))
                })

        all_df = pd.DataFrame(all_records)
        if all_df.empty:
            print("No voxelwise data found.")
            return

        all_df = all_df.explode(['voxels', 'voxel_idx'])
        all_df = all_df.rename(columns={'voxels': 'response'})

        pivot_df = all_df.pivot_table(
            index=['subject', 'hemisphere', 'region', 'voxel_idx'],
            columns=['source', 'feature'],
            values='response'
        ).reset_index()
        pivot_df.columns = pivot_df.columns.map(lambda x: (str(x[0]), str(x[1])) if isinstance(x, tuple) else x)

        def get_feature_values(df, feature, source):
            if isinstance(feature, tuple):
                f1, f2 = feature
                col1 = (source, f1)
                col2 = (source, f2)
                # if col1 in df.columns and col2 in df.columns:
                return df[col1] - df[col2]
                # else:
                    # return pd.Series([np.nan] * len(df))
            else:
                col = (source, feature)
                # if col in df.columns:
                return df[col]
                # else:
                #     return pd.Series([np.nan] * len(df))

        all_results = []
        feature_pairs = list(combinations(feature_names, 2))

        for feature1, feature2 in feature_pairs:
            source1 = feature_sources[feature1]
            source2 = feature_sources[feature2]

            df = pivot_df.copy()
            resp1 = get_feature_values(df, feature1, source1)
            resp2 = get_feature_values(df, feature2, source2)
            
            # #clip negative responses to zero
            # resp1[resp1<0]=0
            # resp2[resp2<0]=0
            
            # Apply joint threshold mask
            mask = (resp1 > 0.0) & \
                   (resp2 > 0.0)

            df = df.loc[mask].assign(response1=resp1[mask], response2=resp2[mask])


            # Ensure both columns are added, even if entirely NaN
            df = df.assign(response1=resp1, response2=resp2)

            pair_label = f"{feature1[0]}-{feature1[1]}" if isinstance(feature1, tuple) else feature1
            pair_label += "_vs_"
            pair_label += f"{feature2[0]}-{feature2[1]}" if isinstance(feature2, tuple) else feature2

            df['comparison'] = pair_label
            all_results.append(df)

        if not all_results:
            print("No valid comparisons found.")
            return

        combined_df = pd.concat(all_results, ignore_index=True)

        if plot_scatter:
            publication = figure_width_mm is not None
            _update_font_rc(9)
            if axis_labels is None:
                axis_labels = ("Image product measure ($R^2$)", "Language product measure ($R^2$)")
            point_size = 3 if publication else 10
            # thousands of voxels: rasterize the points at publication size so the pdf stays small (text stays editable)
            point_kws = dict(linewidth=0, rasterized=True) if publication else {}
            if split_hemisphere:
                col_order = ['MT','pSTS','aSTS','pTemp','aTemp','frontal']
                row_order = ['left', 'right']
                g = sns.FacetGrid(combined_df, col="region", row="hemisphere",
                                  col_order=col_order, row_order=row_order,
                                  sharex=True, sharey=True)
                g.map_dataframe(sns.scatterplot, x="response1", y="response2", alpha=0.35, s=point_size, **point_kws)
                g.map_dataframe(sns.histplot, x="response1", y="response2", bins=20, pthresh=0.1, cmap='rocket', alpha=0.5)
                g.set_axis_labels(*axis_labels, fontsize=_fs(9))
                g.set_titles("{row_name} {col_name}")
                suffix = "_by_hemisphere"
            else:
                g = sns.FacetGrid(combined_df, col="region", col_wrap=3, col_order = ['MT','pSTS','aSTS','pTemp','aTemp','frontal'], sharex=True, sharey=True)
                g.map_dataframe(sns.scatterplot, x="response1", y="response2", alpha=0.35, s=point_size, **point_kws, hue='hemisphere', palette='Accent')
                g.map_dataframe(sns.histplot,x="response1", y="response2", bins=20, pthresh=0.1, cmap='rocket',alpha=0.5)
                g.set_axis_labels(*axis_labels, fontsize=_fs(9))
                g.set_titles("{col_name}")
                g.add_legend()
                if publication:
                    for handle in g._legend.legend_handles if hasattr(g._legend, 'legend_handles') else g._legend.legendHandles:
                        handle.set_sizes([10])
                suffix = ""

            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir + f"{file_tag}scatter_voxelwise_comparisons{suffix}"
            if publication:
                save_path += '.pdf'
                save_figure(g.figure, save_path)
            else:
                save_path += '.png'
                g.savefig(save_path, bbox_inches='tight',dpi=300)
            print(f"Saved scatterplot figure to {save_path}")
            plt.close()

            if save_source_data:
                plain_labels = [label.replace('$R^2$', 'R2') for label in axis_labels]
                source_df = combined_df[['subject', 'hemisphere', 'region', 'voxel_idx', 'response1', 'response2']]
                source_df.columns = ['subject', 'hemisphere', 'region', 'voxel', plain_labels[0], plain_labels[1]]
                source_df = source_df.dropna()
                source_path = os.path.splitext(save_path)[0] + '_source_data.csv'
                source_df.to_csv(source_path, index=False)
                print(f"Saved scatterplot source data to {source_path}")


        # r_values = []
        # grouped = combined_df.groupby(['subject', 'hemisphere', 'region', 'comparison'])
        # for (subject, hemi, region, comparison), group in grouped:
        #     # group = group.dropna(subset=['response1', 'response2'])
        #     if len(group) > 2:
        #         r, _ = pearsonr(group['response1'], group['response2'])
        #         r_values.append({
        #             'subject': subject,
        #             'hemisphere': hemi,
        #             'region': region,
        #             'comparison': comparison,
        #             'r': r
        #         })

        # r_df = pd.DataFrame(r_values)

        # # for hemi in r_df['hemisphere'].unique():
        # #     matrix_data = r_df[r_df['hemisphere'] == hemi].pivot_table(
        # #         index='region', columns='comparison', values='r', aggfunc='mean')

        # #     plt.figure(figsize=(12, 6))
        # #     sns.heatmap(matrix_data, annot=True, cmap='vlag', center=0.0)
        # #     plt.title(f'Pairwise Pearson r (hemisphere: {hemi})')
        # #     plt.xlabel('Comparison')
        # #     plt.ylabel('Region')
        # #     plt.xticks(rotation=90)
        # #     plt.tight_layout()

        # #     if save_dir is not None:
        # #         save_path = os.path.join(save_dir, f"r_matrix_{hemi}.png")
        # #         plt.savefig(save_path, bbox_inches='tight')
        # #         print(f"Saved matrix plot to {save_path}")
        # #     else:
        # #         plt.show()
        
        # # plt.figure(figsize=(30, 6))
        # # hue_order = ['alexnet_vs_motion','alexnet_vs_sbert','motion_vs_sbert','word2vec_vs_sbert']
        # hue_order = [ 'alexnet_vs_'+sbert for sbert in ['sbert_'+layer for layer in ['early','mid','late'] ] ]
        # # hue_order = None#['alexnet_vs_sbert']
        # # hue_order = []
        # hue_order.append('alexnet_vs_motion')
        # # hue_order.append('word2vec_vs_sbert')
        # # hue_order.append('alexnet_vs_sbert')
        
        # g = sns.catplot( kind='bar',
        #     data=r_df,
        #     x='hemisphere', y='r', row='region',row_order=region_names, hue='comparison', hue_order=hue_order,
        #     palette='plasma', linewidth=4, edgecolor='black',
        #     errorbar='se', aspect=4
        # )
        # g.set_titles("{row_name}")
        # g.set_ylabels('voxel-wise product measure correlation')
        # g.set_xlabels('')
        # # plt.legend(title='')
        # # plt.xticks(rotation=45)
        # # plt.tight_layout()
        
        # if save_dir is not None:
        #     os.makedirs(save_dir, exist_ok=True)
        #     save_path =save_dir+"bar_voxelwise.png"
        #     plt.savefig(save_path, bbox_inches='tight',dpi=300)
        #     print(f"Saved figure to {save_path}")
        # else:
        #     plt.show()
def plot_hemispheric_distribution(
    self,
    results,
    column_group,
    col_order,
    loc_name,
    hue_key,
    value_key='proportion_voxels',
    width_ratios=None,
    palette=None,
    restrict_to_feature=None,
    show_box=True,
    save=True,
    file_tag=''
):
    """
    Plot voxel hemispheric distribution for each ROI.

    Parameters
    ----------
    results : pd.DataFrame
        Preprocessed dataframe with 'mask', 'hemisphere', 'subject', and `value_key`.
    column_group : str
        Column to facet by (e.g., 'localizer_contrast_label').
    col_order : list of str
        Ordered list of column_group values.
    loc_name : str
        Column for localizer name (e.g., 'localizer_contrast').
    hue_key : str
        Condition to color by (e.g., 'Feature Space' or 'Model').
    value_key : str
        Variable to plot (default: 'proportion_voxels').
    restrict_to_feature : str or None
        If specified, filter to only this feature (e.g., 'sbert').
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': fontsize, 'font.family': fontfamily})
    df = results.copy()

    if restrict_to_feature:
        df = df[df[hue_key] == restrict_to_feature]

    # Only use right hemisphere to reduce duplication
    df = df[df['hemisphere'] == 'right']
    df = df.sort_values(by='hemi_mask_ID', ascending=False)

    row_order = col_order#[::-1]  # bottom-to-top plotting order

    params = {
        'x': value_key,
        'y': 'mask',
        'hue': loc_name,
        'orient': 'h',
        'palette': palette if palette else self.colors_dict
    }

    g = sns.catplot(
        kind="swarm",
        data=df,
        row=column_group,
        row_order=row_order,
        height=2.7,
        aspect=3.75,
        size=8,
        edgecolor="black",
        linewidth=1.75,
        dodge=False,
        legend=False,
        sharey=False,
        sharex=True,
        **params
    )
    plt.subplots_adjust(wspace=0.1)
    g.set_titles("{row_name}")

    # Optional formatting per axis
    for ax_n in g.axes:
        for ax in ax_n:
            value = ax.title.get_text()#
            label_text = value.split(' (')[0] +' region'
            # Add rectangle and text
            bbox = dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='none', linewidth=1.5)
            text_x = 0.5
            text_y = 1.15
            ax.text(text_x, text_y, label_text, fontsize=20, fontweight='normal',
                    ha='center', va='center', transform=ax.transAxes, bbox=bbox)
            ax.set_title('')
            ax.axvspan(0.5, 1.0, alpha=0.4, facecolor='gray', ec='black', lw=2)
            ax.axvspan(0.0, 0.5, alpha=0.1, facecolor='gray', ec='black', lw=2)

            temp_data = df[df[column_group] == value]
            if show_box:
                sns.boxplot(
                    data=temp_data,
                    x=value_key,
                    y='mask',
                    ax=ax,
                    color='white',
                    saturation=0
                )

    g.set_axis_labels("hemispheric distribution of the most selective voxels\n(more in left $\longleftrightarrow$ more in right)", "")
    g.set(xlim=(-0.02, 1.02))

    if save:
        filename = f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_tag}_hemi_proportion.png"
        filepath = f"{self.figure_dir}/{filename}"
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
def plot_hemispheric_distribution_correlations(
    self,
    results,
    value_col='proportion_voxels',
    index_col='subject',
    region_col='mask',
    file_label='glm_localizer',
    suffix='correlations',
    subset_regions=None  # optional list of region names to subset
):
    """
    Generate a pairwise scatter matrix of voxel distribution correlations across ROIs.

    Parameters
    ----------
    results : pd.DataFrame
        Should contain ['subject', 'mask', 'proportion_voxels'] columns.
    value_col : str
        Column to pivot and correlate (default: 'proportion_voxels').
    index_col : str
        Column for subject IDs (default: 'subject').
    region_col : str
        Column for ROI identity (e.g., 'mask' or 'hemi_mask').
    file_label : str
        Used for output filename.
    suffix : str
        File suffix (default: 'correlations').
    subset_regions : list of str or None
        If specified, restrict plot to this set of regions.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats
    import numpy as np
    plt.rcParams.update({'font.size': 14, 'font.family': fontfamily})
    hemi_data = results[results['hemisphere'] == 'right'].copy()
    # Pivot to subjects × regions
    corr_data = pd.pivot_table(
        data=hemi_data,
        values=value_col,
        columns=[region_col],
        index=[index_col]
    )
    if subset_regions:
        corr_data = corr_data[subset_regions]

    # Define custom annotator and axis remover
    def corrfunc(x, y, **kws):
        nas = np.logical_or(np.isnan(x), np.isnan(y))
        corr, p = scipy.stats.pearsonr(x[~nas], y[~nas])
        ax = plt.gca()
        ax.annotate(f"r = {corr:.2f}", xy=(0.1, 0.9), xycoords=ax.transAxes)
        ax.annotate(f"p = {p:.2f}", xy=(0.1, 0.82), xycoords=ax.transAxes)

    def remove_axes(x, y, **kws):
        ax = plt.gca()
        ax.set(xticks=[], yticks=[])
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

    # Create grid
    g = sns.PairGrid(corr_data)
    g.map_lower(sns.scatterplot)
    g.map_lower(corrfunc)
    g.map_upper(remove_axes)

    # Save
    fname = f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_{suffix}.png"
    fpath = f"{self.figure_dir}/{fname}"
    plt.savefig(fpath, bbox_inches='tight', dpi=300)
    plt.close()
# Display names for the feature space whose weights are being correlated.
SIMILARITY_TITLE_DICT = {
    'alexnet': "Image (AlexNet)",
    'sbert': 'sBERT',
    'hubert': 'Speech (HuBERT)',
    'GPT2': 'Sentence (GPT-2)',
    'gpt2': 'Language (GPT-2)',
    'GPT2_chunk120': 'Sentence (GPT-2)',
    'gpt2_chunk120': 'Language (GPT-2)',
    'motion': 'Motion (pymoten)',
}


def similarity_region_label(mask_label, newline=True):
    """'interact-no_interact-STS' -> 'SI\n(STS)', or 'SI (STS)' on one line."""
    parts = mask_label.split('-')
    contrast = '-'.join(parts[:-1])
    parcel = parts[-1]
    if parcel == 'frontal_language':
        parcel = 'frontal'
    localizer = {
        'interact&no_interact': 'motion',
        'interact-no_interact': 'SI',
        'intact-degraded': 'language',
    }.get(contrast)
    if localizer is None:
        # Unknown contrast: fall back to the raw contrast, run straight into the
        # parcel exactly as this label has always been built.
        return f"{contrast}({parcel})"
    return f"{localizer}\n({parcel})" if newline else f"{localizer} ({parcel})"


def similarity_superregion(mask_label):
    """Parcel of a mask label, collapsed the way run_similarity_superregion_LME does."""
    parcel = mask_label.split('-')[-1]
    if parcel in ('pSTS', 'aSTS', 'STS'):
        return 'STS'
    if parcel in ('pTemp', 'aTemp', 'temporal'):
        return 'temporal'
    return parcel


def plot_similarity_bars(results_df, names, axis, output_path=None, color=None,
                         ylabel='response similarity (r)', ylim=(0, 1),
                         show_points=False, contrasts=None):
    """One bar per region pair: the bar-chart counterpart of plot_similarity_matrix.

    Each subject contributes one correlation per pair (averaged over hemispheres
    where they were kept apart), and the bars show the mean with SEM error bars.

    ``contrasts`` is the pairwise-comparison frame from
    ``stats_helpers.run_similarity_superregion_LME``; pass it to annotate the
    comparisons between the plotted bars.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({'font.size': (fontsize-2)*font_scale, 'font.family': fontfamily})

    records = []
    order = []
    tokens = {}
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            pair = results_df[
                (results_df.axis == axis) &
                (results_df.hemi1 == results_df.hemi2) &
                (((results_df.mask1_name == name1) & (results_df.mask2_name == name2)) |
                 ((results_df.mask1_name == name2) & (results_df.mask2_name == name1)))
            ]
            if pair.empty:
                continue
            # Stacked over three lines: side by side, 'language (temporal)' from
            # one pair runs into the next pair's label.
            label = (f"{similarity_region_label(name1, newline=False)}\n–\n"
                     f"{similarity_region_label(name2, newline=False)}")
            order.append(label)
            tokens[label] = '__'.join(sorted([similarity_superregion(name1),
                                              similarity_superregion(name2)]))
            for subject, value in pair.groupby('subject')['corr'].mean().items():
                records.append({'pair': label, 'subject': subject, 'r': value})

    df = pd.DataFrame(records)
    if df.empty:
        print(f"[plot_similarity_bars] no rows for axis {axis!r}; nothing to plot")
        return df

    fig, ax = plt.subplots(figsize=(2.9*len(order) + 1.2, 5))
    sns.barplot(
        data=df, x='pair', y='r', order=order, ax=ax,
        color=color, saturation=0.75, width=bar_width,
        edgecolor='black', linewidth=2.5,
        errorbar='se', errcolor='black', errwidth=2.5, capsize=0.15,
    )
    if show_points:
        sns.stripplot(data=df, x='pair', y='r', order=order, ax=ax,
                      color='black', alpha=0.35, size=4, jitter=0.12)

    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.spines[['top', 'right']].set_visible(False)

    if contrasts is not None and len(order) > 1:
        # The LME is fit over every region pair in the data, so these p-values
        # are the same bonferroni-corrected ones the paper's table reports --
        # the correction spans pairs that may not be drawn here.
        pairs, pvals = [], []
        for i, label1 in enumerate(order):
            for label2 in order[i+1:]:
                token1, token2 = tokens[label1], tokens[label2]
                row = contrasts[contrasts['contrast'].isin(
                    [f"{token1} - {token2}", f"{token2} - {token1}"])]
                if row.empty:
                    continue
                pairs.append((label1, label2))
                pvals.append(float(row['p.value'].iloc[0]))
        if pairs:
            apply_stat_annotations(
                ax, pairs, pvals, df,
                params={'x': 'pair', 'y': 'r', 'order': order},
            )

    label_text = SIMILARITY_TITLE_DICT.get(axis, axis)
    ax.text(0.5, 1.04, label_text, fontsize=(fontsize-2)*font_scale, fontweight='normal',
            color=get_colors_dict()[axis], ha='center', va='bottom',
            transform=ax.transAxes)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    return df


def plot_similarity_matrix(results_df, names, axis, label_dict, split_hemi=True,output_path=None, plot_cbar=True, cmap=None, vmax=1.0):
    """When figure_fontsize is set, the matrix is saved as an editable pdf (output_path's extension is
    replaced by .pdf) with figure_fontsize text, a title one point larger, vector cells, and the matrix drawn
    at similarity_matrix_mm wide, so matrices saved separately have identical cells when composed.
    plot_cbar=False then leaves out the colorbar and its space (for a colorbar shared across panels)."""
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    publication = figure_fontsize is not None

    plt.rcParams.update({'font.size': fontsize*font_scale, 'font.family': fontfamily})

    if cmap is None:
        cmap = get_cmaps()['matrix_green']

    if(split_hemi):
        hemi_names = ['left', 'right']
    else:
        hemi_names = ['both']
    full_names = [f"{hemi}:{name}" for hemi in hemi_names for name in names]
    similarity_matrix = np.full((len(full_names), len(full_names)), np.nan)

    # tracker = set()

    for i1, label1 in enumerate(full_names):
        for i2, label2 in enumerate(full_names):
            hemi1, mask1_name = label1.split(':')
            hemi2, mask2_name = label2.split(':')

            # search both orderings
            temp1 = results_df[
                (results_df.axis == axis) &
                (results_df.hemi1 == hemi1) & (results_df.mask1_name == mask1_name) &
                (results_df.hemi2 == hemi2) & (results_df.mask2_name == mask2_name)
            ]

            temp2 = results_df[
                (results_df.axis == axis) &
                (results_df.hemi1 == hemi2) & (results_df.mask1_name == mask2_name) &
                (results_df.hemi2 == hemi1) & (results_df.mask2_name == mask1_name)
            ]

            combined = pd.concat([temp1, temp2])
            value = np.nanmean(combined['corr'])

            similarity_matrix[i1, i2] = value
            similarity_matrix[i2, i1] = value

    # plot
    if split_hemi:
        figsize = (15,15)
        label = 'Average Spearman correlation'
        temp_fontsize=fontsize*font_scale
    else:
        figsize = (5,5)
        label = 'r'
        temp_fontsize = (fontsize-2)*font_scale
    title_fontsize = temp_fontsize
    if publication:
        _update_font_rc(temp_fontsize)
        temp_fontsize = figure_fontsize
        title_fontsize = min(figure_fontsize + 1, 7)
    else:
        plt.rcParams.update({'font.size': temp_fontsize, 'font.family': fontfamily})
    similarity_matrix[np.triu_indices(similarity_matrix.shape[0],0)] = np.nan
    cropped_matrix = similarity_matrix[1:, :-1]
    if publication:
        # matrix axes at a fixed physical size on an oversized canvas; the saved pdf is cropped to its contents
        canvas = 6
        size = similarity_matrix_mm / 25.4
        fig = plt.figure(figsize=(canvas, canvas))
        ax = fig.add_axes([1 / canvas, 1 / canvas, size / canvas, size / canvas])
        n_rows, n_cols = cropped_matrix.shape
        # vector cells (an embedded 3x3 image gets smoothed by some pdf viewers)
        cax = ax.pcolormesh(np.arange(n_cols + 1) - 0.5, np.arange(n_rows + 1) - 0.5,
                            np.ma.masked_invalid(cropped_matrix), cmap=cmap, vmin=0, vmax=vmax, linewidth=0)
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.tick_params(length=0)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.imshow(cropped_matrix, cmap=cmap, vmin=0, vmax=vmax)
    sns.despine(left=True, bottom=True)

    if publication and plot_cbar:
        bar_axes = fig.add_axes([(1 + size + 0.08) / canvas, (1 + size * 0.2) / canvas, 0.07 / canvas, size * 0.6 / canvas])
        cbar = fig.colorbar(cax, cax=bar_axes, label=label)
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(width=0.5, length=2)
    elif publication:
        pass
    elif plot_cbar:
        # The label sits on the colorbar's *left*, i.e. in the gap between it and
        # the matrix, so the default pad lets it land on the top-right cell
        # whenever that cell is filled. Widen the gap to keep it clear.
        cbar = fig.colorbar(cax, label=label, shrink=0.5, pad=0.12)
        cbar.ax.yaxis.set_label_position('left')
    else:
        cbar = fig.colorbar(cax)
        cbar.ax.set_visible(False)

    ##### Build clean axis labels
    label_names = [similarity_region_label(full_label.split(':')[1])
                   for full_label in full_names]

    cropped_labels_y = label_names[1:]
    cropped_labels_x = label_names[:-1]
    ax.set_xticks(range(len(cropped_labels_x)))
    ax.set_yticks(range(len(cropped_labels_y)))
    ax.set_xticklabels(cropped_labels_x, rotation=0)
    ax.set_yticklabels(cropped_labels_y)
    plt.title('')
    label_text= SIMILARITY_TITLE_DICT.get(axis,axis)
    fig.subplots_adjust(top=0.75)  
    # Add rectangle and text
    text_x = 0.45
    text_y = 1.05
    ax.text(text_x, text_y, label_text, fontsize=title_fontsize,fontweight='normal', color = get_colors_dict()[axis],
            ha='center', va='center', transform=ax.transAxes,clip_on=False)

    # add text values
    for i in range(len(cropped_labels_x)):
        for j in range(len(cropped_labels_y)):
            val = cropped_matrix[j, i]
            if not np.isnan(val):
                ax.text(i, j, f"{val:.2f}", ha='center', va='center', fontsize=temp_fontsize)

    if(split_hemi):
        # hemisphere divider
        n = len(cropped_labels_y) // 2
        ax.plot([-0.5, n+0.5], [n-0.5, n-0.5], color='black', linewidth=2)
        ax.plot([n+0.5, n+0.5], [n-0.5, len(cropped_labels_y)-0.5], color='black', linewidth=2)
        
        # hemisphere labels
        ax.text(-2.5, n / 2 - 0.5, 'Left', ha='left', va='center', fontweight='normal')
        ax.text(-2.5, n + (n + 1) / 2 - 0.5, 'Right', ha='left', va='center', fontweight='normal')
        ax.text(n / 2 - 0.5, len(cropped_labels_y) + 0.5, 'Left', ha='center', va='bottom', fontweight='normal')
        ax.text(n + (n+1) / 2 - 0.5, len(cropped_labels_y) + 0.5, 'Right', ha='center', va='bottom', fontweight='normal')

    if output_path and publication:
        fig.savefig(os.path.splitext(output_path)[0] + '.pdf', bbox_inches='tight', pad_inches=0.02)
    elif output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    return similarity_matrix, full_names
def plot_top_unit_scores(csv_dir,model,layer_to_plot='alexnet_layer7',y='score',col_wrap=None,hue_order=None,selected_units=None,top_n=None):
    # import glob as glob

    def load_and_prepare_annotation_corrs(csv_dir,model,layer_to_plot):

        import pandas as pd
        
        if(y=='correlation'):
            all_df = pd.read_csv(f"{csv_dir}/subjectwise_unitwise_annotation_corrs_{model}_{layer_to_plot}.csv")
            mask_label = 'mask'
        elif(y=='score'):
            all_df = pd.read_csv(f"{csv_dir}/subjectwise_unit_feature_CCA_{model}_{layer_to_plot}_{top_n}.csv")
            mask_label = 'region'
         # Map region to superregion
        # print(all_df)
        region_map = {
            "MT": "motion", "STS": "social interaction",
            "temporal": "language", "frontal": "language"
        }
        
        all_df["superregion"] = all_df[mask_label].replace(region_map)
        
        # hemi_region_map = {
        #     "left_MT":"left MT", "right_MT":"right MT",
        #     "left_pSTS": "left STS", "left_aSTS": "left STS",
        #     "right_pSTS": "right STS", "right_aSTS": "right STS",
        #     "left_pTemp": "left language", "left_aTemp": "left language",
        #     "right_pTemp": "right language", "right_aTemp": "right language"
        # } 
        # all_df["superhemiregion"] = all_df["hemi_region"].replace(hemi_region_map)

        # # Melt into longform
        # long_df = all_df.melt(
        #     id_vars=["subject", "feature", "superhemiregion", "superregion","layer"],
        #     var_name="annotated_feature",
        #     value_name="correlation"
        # )
        # print(long_df)
        
        if(selected_units!=None):
            all_df = all_df[all_df['unit_index'].isin(selected_units)]
        # print(all_df)
            
        # First: average across region → superregion per subject
        group_by = ['subject',mask_label,'superregion','annotated_feature']
        if('hemi' in  all_df.columns):
            group_by.append('hemi')
        grouped = (
            all_df
            .groupby(group_by, as_index=False)
            .mean()
        )
        
        order_dict = {
            'MT': 0, 'STS': 1, 'temporal':2,'frontal':3
        }
        grouped['mask_label_ID'] = [order_dict[x] for x in grouped[mask_label]]
        grouped = grouped.sort_values(by='mask_label_ID', ascending=True)

        
        

        return grouped,mask_label
    
    df,mask_label = load_and_prepare_annotation_corrs(csv_dir, model, layer_to_plot)
    
    # df  = df[df['layer']==layer_to_plot]
    df['mask'] = df[mask_label]
    df['region'] = df['superregion']
    df['Annotated Feature'] = df['annotated_feature']
    
    x = 'mask'
    if('hemi' in df.columns):
        x='hemi'

    # Plotting
    return plot_bar_and_strip(
        data=df,
        column_group="superregion",
        col_order=['motion','social interaction','language'],
        col_wrap=col_wrap,
        width_ratios=[1,1,2],#[1.0] * len(df["region"].unique()),
        params={
            "x": x,
            "y": y,
            "hue": "Annotated Feature",
            "hue_order": hue_order, #+ ['word2vec','motion'],
            "palette": get_colors_dict()
        },
        height=5,
        aspect=1.2
    )
def create_unit_montages(
    unit_image_dict,
    output_dir,
    image_size=(160, 90),
    spacing=10,
    font_path=None
):
    """
    Create montage images showing top and bottom images for each unit.

    Parameters
    ----------
    unit_image_dict : dict
        Output of get_top_and_bottom_images_per_unit. Keys are unit indices.
        Each value is a dict with 'top' and 'bottom' lists of image paths.
    output_dir : str
        Directory to save montages.
    image_size : tuple[int, int]
        Size to resize each image.
    spacing : int
        Padding between images.
    font_path : str or None
        Path to a .ttf font for labeling. Uses default if None.
    """
    from PIL import Image, ImageDraw, ImageFont
    import os

    os.makedirs(output_dir, exist_ok=True)

    try:
        font = ImageFont.truetype(font_path, 16) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    label_height = 20
    text_offset = 5

    for unit, imgs in unit_image_dict.items():
        top_imgs = [Image.open(path).resize(image_size) for path in imgs['top']]
        bottom_imgs = [Image.open(path).resize(image_size) for path in imgs['bottom']]

        n = len(top_imgs)
        img_w, img_h = image_size

        width = n * img_w + (n + 1) * spacing
        height = (
            spacing * 4 +
            label_height +  # Unit title
            label_height +  # Top label
            img_h +
            label_height +  # Bottom label
            img_h
        )

        montage = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(montage)

        # Draw unit title at top left
        draw.text((spacing, spacing), f"Unit {unit}", fill="black", font=font)

        # Position for top label and images
        y_top_label = spacing + label_height
        y_top_imgs = y_top_label + label_height + text_offset

        # Position for bottom label and images
        y_bottom_label = y_top_imgs + img_h + spacing
        y_bottom_imgs = y_bottom_label + label_height + text_offset

        draw.text((spacing, y_top_label), "Top activations", fill="black", font=font)
        draw.text((spacing, y_bottom_label), "Bottom activations", fill="black", font=font)

        for i in range(n):
            x = spacing + i * (img_w + spacing)
            montage.paste(top_imgs[i], (x, y_top_imgs))
            montage.paste(bottom_imgs[i], (x, y_bottom_imgs))

        montage_path = os.path.join(output_dir, f"unit_{unit}_montage.jpg")
        montage.save(montage_path)
def apply_stat_annotations(
    ax,
    pairs,
    pvals,
    data,
    params,
    text_format='star',
    hide_non_significant=True,
    fontsize='x-small',
    verbose=False,
    test=None
):
    """
    Apply statistical annotations to a Seaborn/Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    pairs : list of tuple
        Pairwise comparisons (e.g., [(group1, group2), ...]).
    pvals : list of float
        FDR or Bonferroni-corrected p-values.
    data : pd.DataFrame
        Data passed to Annotator.
    params : dict
        Plotting params (at least must include 'x', 'y', and 'hue').
    Other parameters control appearance.
    """
    from statannotations.Annotator import Annotator
    annot = Annotator(ax, pairs, data=data, verbose=verbose, **params)
    annot.configure(
        test=test,  # test=None just uses given p-values
        text_format=text_format,
        show_test_name=False,
        hide_non_significant=False, #hide_non_significant,
        fontsize=_fs(fontsize), #'x-small' would fall below 5 pt at publication size
        line_width=_lw(1.5)
    )
    annot.set_pvalues(pvals)
    annot.annotate()