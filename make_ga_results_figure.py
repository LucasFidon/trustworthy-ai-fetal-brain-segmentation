import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from src.utils.definitions import *

CSV_RES = '/data/saved_res_fetal_trust21_v3/nnunet_task225/metrics.csv'
PLOT_SIZE = [12, 8]
SNS_FONT_SCALE = 3.0
GA = [23, 32]
ROI_NAMES_TO_DISPLAY = {
    'white_matter': 'White Matter',
    'intra_axial_csf': 'Intra-axial CSF',
    'cerebellum': 'Cerebellum',
    'extra_axial_csf': 'Extra-axial CSF',
    'cortical_grey_matter': 'Cortical Grey Matter',
    'deep_grey_matter': 'Deep Gray Matter',
    'brainstem': 'Brainstem',
    'corpus_callosum': 'Corpus Callosum',
}
METHOD_NAME_TO_DISPLAY = {
    'cnn': 'AI',
    'atlas': 'Fallback',
    'trustworthy': 'Trustworthy AI',
}
METRIC_NAME_TO_DISPLAY = {
    'dice': 'Dice Score (in %)',
    'hausdorff': 'Hausdorff dist. (in mm)',
}

def main(metric):
    sns.set(font_scale=SNS_FONT_SCALE)
    sns.set_style("whitegrid")

    df = pd.read_csv(CSV_RES)

    # Filter consition
    df = df[df['Condition'] != 'Pathological']

    # Filter GA
    df = df[df['GA'] >= GA[0]]
    df = df[df['GA'] <= GA[1]]

    # Change names of methods
    for met in list(METHOD_NAME_TO_DISPLAY.keys()):
        df.loc[df['Methods'] == met, 'Methods'] = METHOD_NAME_TO_DISPLAY[met]

    nrows = 4
    ncols= 2
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(PLOT_SIZE[0] * ncols, PLOT_SIZE[1] * nrows),
    )

    for k, roi in enumerate(ALL_ROI):
        i = k % nrows
        j = k // nrows
        df_roi = df[df['ROI'] == roi]
        sns.lineplot(
            data=df_roi,
            y=metric,
            x='GA',
            hue='Methods',
            style='Methods',
            markers=True,
            dashes=False,
            ax=ax[i,j],
        )
        # Title
        ax[i,j].set_title(ROI_NAMES_TO_DISPLAY[roi], fontweight='bold')
        # X-axis title
        if i == 3:
            ax[i,j].set_xlabel('Gestational Ages', fontweight='bold')
        else:
            ax[i,j].set(xlabel=None)
        if j == 0:
            ax[i,j].set_ylabel(METRIC_NAME_TO_DISPLAY[metric], fontweight='bold')
        else:
            ax[i,j].set(ylabel=None)
        # Legend
        if i != 3 or j !=0:
            ax[i,j].get_legend().remove()

    fig.suptitle('Mean and 95%% CI for the %s' % METRIC_NAME_TO_DISPLAY[metric])
    fig.tight_layout()
    save_name = '%s_Control_and_SB_GA.png' % metric
    fig.savefig(save_name, bbox_inches='tight')
    print('Figure saved in', save_name)


if __name__ == '__main__':
    for metric in METRIC_NAMES:
        main(metric)
