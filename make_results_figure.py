import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.transforms as mtrans
import seaborn as sns
from run_infer_eval import SAVE_FOLDER
from src.utils.definitions import *

CSV_RES = '/data/saved_res_fetal_trust21_v3/nnunet_task225/metrics.csv'
PKL_FILES = {
    center: {
        cond: os.path.join(
            SAVE_FOLDER,
            'nnunet_task225',
            'metrics_%s-distribution_%s.pkl' % (center, cond.replace(' ', '_'))
        )
        for cond in CONDITIONS
    }
    for center in CENTERS
}
METHODS_TO_PLOT = ['cnn', 'atlas', 'trustworthy']
METHOD_NAME_TO_DISPLAY = {
    'cnn': 'AI',
    'atlas': 'Fallback',
    'trustworthy': 'Trustworthy AI',
}
ROI_NAMES_TO_DISPLAY = {
    'white_matter': 'WM',
    'intra_axial_csf': 'In-CSF',
    'cerebellum': 'CER',
    'extra_axial_csf': 'Ext-CSF',
    'cortical_grey_matter': 'CGM',
    'deep_grey_matter': 'DGM',
    'brainstem': 'BST',
    'corpus_callosum': 'CC',
}
CONDITION_NAMES_TO_DISPLAY = {
    'Neurotypical': 'Neurotypical',
    'Spina Bifida': 'Spina Bifida',
    'Pathological': 'Other Pathologies',
}
CENTER_TYPES_TO_DISPLAY = {
    'in': 'In-scanner Distribution',
    'out': 'Out-of-scanner Distribution',
}
CENTER_TYPES_TO_DISPLAY_AGGREGATED = {
    'in': 'In-scanner\nDistribution',
    'out': 'Out-of-scanner\nDistribution',
}
METRIC_NAMES_TO_DISPLAY = {
    'dice': 'Dice score (in %)',
    'hausdorff': 'Hausdorff dist. 95% (in mm)',
}
BOXPLOT_SIZE = {
    True: [11, 10],
    False: [15, 10],
}
YAXIS_LIM = {
    'dice': {
        'Neurotypical': (48, 100),
        'Spina Bifida': (-2, 100),
        'Pathological': (-2, 100),
    },
    'hausdorff': {  # Rk: max is 57.6mm
        'Neurotypical': (-0.3, 12.3),
        'Spina Bifida': (-1, 36),
        'Pathological': (-0.5, 18),
    },
}
YAXIS_LIM_AGGREGATED = {
    'dice': {
        'Neurotypical': (69, 100),
        'Spina Bifida': (18, 100),
        'Pathological': (18, 100),
    },
    'hausdorff': {  # Rk: max is 57.6mm
        'Neurotypical': (-0.3, 12.3),
        'Spina Bifida': (-1, 36),
        'Pathological': (-0.5, 18),
    },
}
YTICKS_HD = {
    'Neurotypical': [i*2 for i in range(0, 7)],
    'Spina Bifida': [i*5 for i in range(0, 8)],
    'Pathological': [i*2.5 for i in range(0, 8)],
}
FONT_SIZE_AXIS = 55
SNS_FONT_SCALE = 2.8
LEGEND_POSITION = {
    'dice': 'lower left',
    'hausdorff': 'upper left',
}
VERTICAL_LINE_SHIFT = 0.025  # Cooking here.. shift for the vertical line that separates subplots


def create_df(metric, condition, center_type, average_roi=False):
    df = pd.read_csv(CSV_RES)

    # Filter data
    df = df[df['Condition'] == condition]
    df = df[df['Center type'] == center_type]
    if average_roi:  # Remove CC and average metric across ROIs
        df = df[df['ROI'] != 'corpus_callosum']
        df = df.groupby(['Study', 'GA', 'Condition', 'Center type', 'Methods'])[metric].mean().reset_index()
    else:
        # Rename the ROIs
        for roi in list(ROI_NAMES_TO_DISPLAY.keys()):
            df.loc[df['ROI'] == roi, 'ROI'] = ROI_NAMES_TO_DISPLAY[roi]
        # Clip high values for the Hausdorff distance
        if metric == 'hausdorff':
            max_val = YAXIS_LIM[metric][condition][1] \
                            - 0.01 * (YAXIS_LIM[metric][condition][1] - YAXIS_LIM[metric][condition][0])
            df.loc[df[metric] > max_val, metric] = max_val

    # Change names of methods
    for met in list(METHOD_NAME_TO_DISPLAY.keys()):
        df.loc[df['Methods'] == met, 'Methods'] = METHOD_NAME_TO_DISPLAY[met]

    return df


def main(metric_name, aggregated=False):
    sns.set(font_scale=SNS_FONT_SCALE)
    sns.set_style("whitegrid")
    if aggregated:
        nrows = len(CENTERS)
        ncols = len(CONDITIONS)
    else:
        nrows = len(CONDITIONS)
        ncols = len(CENTERS)
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(BOXPLOT_SIZE[aggregated][0] * ncols, BOXPLOT_SIZE[aggregated][1] * nrows),
    )
    for i, condition in enumerate(CONDITIONS):
        for j, center_type in enumerate(CENTERS):
            df = create_df(metric_name, condition, center_type, average_roi=aggregated)
            # import pdb
            # pdb.set_trace()

            # Create the boxplot
            if aggregated:
                ax_ij = ax[j,i]
                print('*** %s - %s: %d cases' % (condition, center_type, len(df) / 3))
                g = sns.boxplot(
                    data=df,
                    x='Methods',
                    y=metric_name,
                    ax=ax_ij,
                    palette='colorblind',
                    fliersize=10,
                    linewidth=3,
                    order=['AI', 'Fallback', 'Trustworthy AI'],
                )

                # X axis
                if j == nrows - 1:
                    ax_ij.set_xlabel(
                        '\n' + CONDITION_NAMES_TO_DISPLAY[condition],
                        fontsize=FONT_SIZE_AXIS,
                        fontweight='bold',
                    )
                else:
                    ax_ij.set(xlabel=None)

                # Y axis
                # ax_ij.set(ylim=YAXIS_LIM_AGGREGATED[metric_name][condition])
                if i == 0:
                    ax_ij.set_ylabel(
                        CENTER_TYPES_TO_DISPLAY_AGGREGATED[center_type] + '\n',
                        fontsize=FONT_SIZE_AXIS,
                        fontweight='bold',
                    )
                else:
                    ax_ij.set(ylabel=None)

                # # Y ticks
                # if metric_name == 'hausdorff':
                #     g.set(yticks=YTICKS_HD[condition])
                #     yticklabels = [str(i) for i in YTICKS_HD[condition]]
                #     yticklabels[-1] += '+'
                #     g.set(yticklabels=yticklabels)

            else:  # No aggregation
                ax_ij = ax[i,j]
                if center_type == 'out':
                    # no CC
                    order = [ROI_NAMES_TO_DISPLAY[roi] for roi in ALL_ROI[:-1]]
                else:
                    order = [ROI_NAMES_TO_DISPLAY[roi] for roi in ALL_ROI]
                g = sns.boxplot(
                    data=df,
                    hue='Methods',
                    y=metric_name,
                    x='ROI',
                    ax=ax_ij,
                    palette='colorblind',
                    fliersize=10,
                    linewidth=3,
                    hue_order=['AI', 'Fallback', 'Trustworthy AI'],
                    order=order,
                )

                # X axis
                if i == nrows - 1:
                    ax[i,j].set_xlabel(
                        '\n' + CENTER_TYPES_TO_DISPLAY[center_type],
                        fontsize=FONT_SIZE_AXIS,
                        fontweight='bold',
                    )
                else:
                    ax[i,j].set(xlabel=None)

                # Y axis
                ax[i,j].set(ylim=YAXIS_LIM[metric_name][condition])
                if j == 0:
                    ax[i,j].set_ylabel(
                        CONDITION_NAMES_TO_DISPLAY[condition] + '\n' ,
                        fontsize=FONT_SIZE_AXIS,
                        fontweight='bold',
                    )
                else:
                    ax[i,j].set(ylabel=None)

                # Y ticks
                if metric_name == 'hausdorff':
                    g.set(yticks=YTICKS_HD[condition])
                    yticklabels = [str(i) for i in YTICKS_HD[condition]]
                    yticklabels[-1] += '+'
                    g.set(yticklabels=yticklabels)

            # Legend
            if aggregated:
                pass
            elif j == 0 and i == 0:
                sns.move_legend(
                    ax[i,j],
                    LEGEND_POSITION[metric_name],
                    # bbox_to_anchor=(1, 0.),
                )
            else:
                ax[i,j].get_legend().remove()

    # Adjust the margins between the subplots
    fig.subplots_adjust(wspace=0.08, hspace=0.15)

    # Add title
    pre = 'Mean-ROI ' if aggregated else ''
    fig.suptitle(pre + METRIC_NAMES_TO_DISPLAY[metric], fontsize=80)

    # Remove extra empty space
    fig.tight_layout()

    if not aggregated:
        # Add the lines between the subplots
        # Get the bounding boxes of the axes including text decorations
        r = fig.canvas.get_renderer()
        get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        bboxes = np.array(list(map(get_bbox, ax.flat)), mtrans.Bbox).reshape(ax.shape)
        # Get the minimum and maximum extent, get the coordinate half-way between those
        xmax = np.array(list(map(lambda b: b.x1, bboxes.flat))).reshape(ax.shape).max(axis=1)
        xmin = np.array(list(map(lambda b: b.x0, bboxes.flat))).reshape(ax.shape).min(axis=1)
        xs = np.c_[xmax[1:], xmin[:-1]].mean(axis=1)
        ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(ax.shape).max(axis=1)
        ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(ax.shape).min(axis=1)
        ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)
        # Draw the lines
        for x in xs:
            line = plt.Line2D(
                [x + VERTICAL_LINE_SHIFT, x + VERTICAL_LINE_SHIFT],
                [0., 0.94], transform=fig.transFigure, color="black")
            fig.add_artist(line)
        for y in ys:
            line = plt.Line2D([0, 1], [y, y], transform=fig.transFigure, color="black")
            fig.add_artist(line)

    # Save the figure
    if aggregated:
        save_name = '%s_aggregated.png' % metric_name
    else:
        save_name = '%s.png' % metric_name
    fig.savefig(save_name, bbox_inches='tight')
    print('Figure saved in', save_name)


if __name__ == '__main__':
    for metric in METRIC_NAMES:
        for aggregated in [True, False]:
            main(metric, aggregated)
