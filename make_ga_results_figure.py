import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from src.utils.definitions import *

CSV_RES = '/data/saved_res_fetal_trust21_v3/nnunet_task225/metrics.csv'
USE_ABN = False
PLOT_SIZE = {
    False: [14, 8],
    True: [14, 10],
}
SNS_FONT_SCALE = 3.0
if USE_ABN:
    GA = [19, 36]
else:
    GA = [19, 35]
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
    'dice': 'Dice Score',
    'hausdorff': 'Hausdorff Dist.',
}

def main(metric, aggregated=False):
    sns.set(font_scale=SNS_FONT_SCALE)
    sns.set_style("whitegrid")

    df = pd.read_csv(CSV_RES)

    # Filter condition
    if not USE_ABN:
        df = df[df['Condition'] != 'Pathological']
    if aggregated:
        df = df[df['ROI'] != 'corpus_callosum']

    # Filter GA
    df['GA'] = df['GA'].round(decimals=0)
    df = df[df['GA'] >= GA[0]]
    df = df[df['GA'] <= GA[1]]

    # Change names of methods
    for met in list(METHOD_NAME_TO_DISPLAY.keys()):
        df.loc[df['Methods'] == met, 'Methods'] = METHOD_NAME_TO_DISPLAY[met]

    if aggregated:
        nrows = 1
        ncols= 1
    else:
        nrows = 4
        ncols= 2
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(PLOT_SIZE[aggregated][0] * ncols, PLOT_SIZE[aggregated][1] * nrows),
    )

    if aggregated:
        # Average the metric across ROIs
        df_ave = df.groupby(['Study', 'GA', 'Methods'])[metric].mean().reset_index()
        sns.lineplot(
            data=df_ave,
            y=metric,
            x='GA',
            hue='Methods',
            style='Methods',
            markers=False,
            dashes=False,
            ax=ax,
            hue_order=['AI', 'Fallback', 'Trustworthy AI'],
            palette='colorblind',
        )
        # X-axis title
        ax.set_xlim(GA)
        ax.set_xticks(range(GA[0], GA[1]+1))
        ax.set_xlabel('Gestational Ages (in weeks)', fontweight='bold')
        ax.set_ylabel('Mean-ROI ' + METRIC_NAME_TO_DISPLAY[metric], fontweight='bold')
    else:
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
                markers=False,
                dashes=False,
                ax=ax[i,j],
                hue_order=['AI', 'Fallback', 'Trustworthy AI'],
                palette='colorblind',
            )
            # Title
            ax[i,j].set_title(ROI_NAMES_TO_DISPLAY[roi], fontweight='bold')
            # X-axis title
            ax[i,j].set_xlim(GA)
            ax[i,j].set_xticks(range(GA[0], GA[1]+1))
            if i == 3:
                ax[i,j].set_xlabel('Gestational Ages (in weeks)', fontweight='bold')
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
    post = '_aggregated' if aggregated else ''
    if USE_ABN:
        save_name = '%s_GA%s.png' % (metric, post)
    else:
        save_name = '%s_Control_and_SB_GA%s.png' % (metric, post)
    fig.savefig(save_name, bbox_inches='tight')
    print('Figure saved in', save_name)


if __name__ == '__main__':
    for metric in METRIC_NAMES:
        for aggregated in [True, False]:
            main(metric, aggregated)
