import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.transforms as mtrans
import seaborn as sns
from run_infer_eval import SAVE_FOLDER
from src.utils.definitions import *


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
BOXPLOT_SIZE = [15, 10]  # Size of each subplot
FONT_SIZE_AXIS = 40
SNS_FONT_SCALE = 2.8
VERTICAL_LINE_SHIFT = 0.025  # Cooking here.. shift for the vertical line that separates subplots

def load_metrics(metrics_path):
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics

def create_df(metric, condition, center_type):
    raw_data = []

    # Create the columns
    columns = ['Study', 'Methods', 'ROI', metric]

    # Create the raw data
    metrics = load_metrics(PKL_FILES[center_type][condition])
    for method in METHODS_TO_PLOT:
        m = metrics[method]

        # Set the right list of ROIs
        roi_list = ALL_ROI
        if condition == 'Pathological' and center_type == 'in':
            roi_list = ['white_matter', 'intra_axial_csf', 'cerebellum']
        elif center_type == 'out':  # no corpus callosum
            roi_list = [
                'white_matter', 'intra_axial_csf', 'cerebellum',
                'extra_axial_csf', 'cortical_grey_matter', 'deep_grey_matter', 'brainstem'
            ]

        # Get the metric values
        for roi in roi_list:
            metric_name = '%s_%s' % (metric, roi)
            num_cases = len(m[metric_name])
            print('**** %s: %d cases' % (roi, num_cases))
            for i in range(num_cases):
                line_base = [i, METHOD_NAME_TO_DISPLAY[method]]
                metric_name = '%s_%s' % (metric, roi)
                line = line_base + [ROI_NAMES_TO_DISPLAY[roi], m[metric_name][i]]
                raw_data.append(line)

    df = pd.DataFrame(raw_data, columns=columns)
    return df

def main(metric_name):
    sns.set(font_scale=SNS_FONT_SCALE)
    sns.set_style("whitegrid")
    nrows = len(CONDITIONS)
    ncols = len(CENTERS)
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(BOXPLOT_SIZE[0] * ncols, BOXPLOT_SIZE[1] * nrows),
    )
    for i, condition in enumerate(CONDITIONS):
        for j, center_type in enumerate(CENTERS):
            df = create_df(metric_name, condition, center_type)
            sns.boxplot(
                data=df,
                hue='Methods',
                y=metric_name,
                x='ROI',
                ax=ax[i,j],
                # palette='Set3',
                palette='colorblind',
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
            if metric_name == 'dice':
                ax[i,j].set(ylim=(-2, 100))
            if j == 0:
                ax[i,j].set_ylabel(
                    CONDITION_NAMES_TO_DISPLAY[condition] + '\n' ,
                    fontsize=FONT_SIZE_AXIS,
                    fontweight='bold',
                )
            else:
                ax[i,j].set(ylabel=None)

            # Legend
            if j == 0 and i == 0:
                sns.move_legend(
                    ax[i,j],
                    "lower left",
                    # bbox_to_anchor=(1, 0.),
                )
            else:
                ax[i,j].get_legend().remove()

    # Adjust the margins between the subplots
    fig.subplots_adjust(wspace=0.08, hspace=0.15)

    # Remove extra empty space
    fig.tight_layout()

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
            [0, 1], transform=fig.transFigure, color="black")
        fig.add_artist(line)
    for y in ys:
        line = plt.Line2D([0, 1], [y, y], transform=fig.transFigure, color="black")
        fig.add_artist(line)

    # Save the figure
    save_name = '%s.png' % metric_name
    fig.savefig(save_name, bbox_inches='tight')
    print('Figure saved in', save_name)


if __name__ == '__main__':
    for metric in METRIC_NAMES:
        main(metric)
