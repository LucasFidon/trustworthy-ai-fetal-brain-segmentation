import numpy as np
import csv
from matplotlib import pyplot as plt
import seaborn as sns
from src.utils.definitions import *

SPLIT_CENTERS = ['training', 'testing_in', 'testing_out']
CONDITION_NAMES_TO_DISPLAY = {
    'Neurotypical': 'Neurotypical',
    'Spina Bifida': 'Spina Bifida',
    'Pathological': 'Other Pathologies',
}
CENTER_TYPES_TO_DISPLAY = {
    'in': 'In-scanner\nDistribution',
    'out': 'Out-of-scanner\nDistribution',
}
COLOR = {
    'training': 'darkgreen',
    'testing': 'royalblue',
}
BOXPLOT_SIZE = [15, 10]  # Size of each subplot
FONT_SIZE_AXIS = 55
FONT_SIZE_NB_CASES = 45
SNS_FONT_SCALE = 2.8
FETA_EXCLUDED = ['sub-007', 'sub-009']
HARVARD_ATLAS_GA = [i for i in range(21, 39)]
CHINESE_ATLAS_GA = [i for i in range(22, 36)]
SB_ATLAS_GA = [i for i in range(21, 35)] + [25]


def get_ga(condition, center_type, data_split):
    ga_list = []
    pat_list = []

    # Add atlas GAs
    if data_split == 'training':
        if condition == 'Neurotypical':
            ga_list += HARVARD_ATLAS_GA
            ga_list += CHINESE_ATLAS_GA
        elif condition == 'Spina Bifida':
            ga_list += SB_ATLAS_GA

    # Other GAs
    for tsv in [INFO_DATA_TSV, INFO_DATA_TSV2, INFO_DATA_TSV3, INFO_TRAINING_DATA_TSV]:
        first_line = True
        with open(tsv) as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                if first_line:
                    first_line = False
                    continue
                pat_id = line[0]
                if pat_id in FETA_EXCLUDED:  # Two spina bifida excluded
                    continue
                cond = line[1]
                # Get GA
                ga = float(line[2])
                # Get center
                if tsv == INFO_DATA_TSV:
                    center = 'out'
                else:
                    center = line[3]
                # Get Training/Test split
                if tsv == INFO_TRAINING_DATA_TSV:
                    split = 'training'
                    center = 'in'
                else:
                    split = 'testing'
                if cond == condition and center == center_type and split == data_split:
                    ga_list.append(ga)
                    assert not pat_id in pat_list, 'ID %s was found twice.' % pat_id
                    pat_list.append(pat_id)
    print('\n%s - %s - %s' % (condition, center_type, data_split))
    print('%d cases' % len(pat_list))
    # print(pat_list)
    return np.array(ga_list)


def main():
    sns.set(font_scale=SNS_FONT_SCALE)
    sns.set_style("whitegrid")
    nrows = len(CONDITIONS)
    ncols = len(SPLIT_CENTERS)
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(BOXPLOT_SIZE[0] * ncols, BOXPLOT_SIZE[1] * nrows),
    )
    for j, condition in enumerate(CONDITIONS):
        for i, split_center in enumerate(SPLIT_CENTERS):
            if split_center == 'training':
                split = 'training'
                center_type = 'in'
            else:
                split, center_type = split_center.split('_')
            ga = get_ga(condition, center_type, split)
            if ga.size > 0:
                sns.distplot(
                    ga,
                    kde=False,
                    rug=False,
                    norm_hist=False,
                    bins=range(19, 40),
                    hist_kws=dict(edgecolor="black", linewidth=3),
                    color=COLOR[split],
                    ax=ax[i,j],
                )

            if ga.size > 0:
                ax[i,j].set_title(
                    'Total: %d 3D MRIs' % ga.size,
                    fontsize=FONT_SIZE_NB_CASES,
                )

            # X axis
            ax[i,j].set_xlim((19, 40))
            if j == 0:
                column_name =''
                if split == 'training':
                    column_name += 'Training\n'
                    column_name += CENTER_TYPES_TO_DISPLAY[center_type] + '\n'
                else:
                    column_name += 'Testing\n'
                    column_name += CENTER_TYPES_TO_DISPLAY[center_type] + '\n'
                ax[i,j].set_ylabel(
                    column_name,
                    fontsize=FONT_SIZE_AXIS,
                    fontweight='bold',
                )
            else:
                ax[i,j].set(ylabel=None)

            # Y axis
            if i == nrows - 1:
                ax[i,j].set_xlabel(
                    '\n' + CONDITION_NAMES_TO_DISPLAY[condition],
                    fontsize=FONT_SIZE_AXIS,
                    fontweight='bold',
                )
            else:
                ax[i,j].set(xlabel=None)
            if ga.size == 0:
                ax[i,j].set_ylim((0, 6))

    fig.suptitle(
        'Histograms of Number of 3D MRIs per Gestational Age (in weeks)',
        fontsize=FONT_SIZE_AXIS+5,
    )
    # Adjust the margins between the subplots
    fig.subplots_adjust(wspace=0.08, hspace=0.15)

    # Remove extra empty space
    fig.tight_layout()

    # Save the figure
    save_name = 'ga_histograms.png'
    fig.savefig(save_name, bbox_inches='tight')
    print('Figure saved in', save_name)


if __name__ == '__main__':
    main()
