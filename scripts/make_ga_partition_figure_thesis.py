import numpy as np
import os
import csv
from matplotlib import pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.definitions import *

# SPLIT_CENTERS = ['training', 'testing_in', 'testing_out']
SPLIT_CENTERS = [
    'FeTA',#
    'Atlases',
    'UHL',
    'UCLH',
    'KCL',
    'MUV',
    'UK',
]
CONDITION_NAMES_TO_DISPLAY = {
    'Neurotypical': 'Neurotypical',
    'Spina Bifida': 'Spina Bifida',
    'Pathological': 'Other Pathologies',
}
CENTER_TO_DISPLAY = {
    'FeTA': 'FeTA dataset\n(Switzerland)',
    'UHL': 'University Hospital\nLeuven',
    'UCLH': 'University College\nLondon Hospital',
    'KCL': 'King\'s College\nLondon',
    'MUV': 'Medical University\nof Vienna',
    'Atlases': 'Atlases\n (UK, US, and China)',
    'UK': 'Various\nhospitals UK',
}
COLOR = {
    'training': 'darkgreen',
    'testing': 'royalblue',
}
BOXPLOT_SIZE = [15, 10]  # Size of each subplot
FONT_SIZE_AXIS = 60
FONT_SIZE_NB_CASES = 55
FONT_SIZE_TITLE = 95
SNS_FONT_SCALE = 4.0  # 2.8
FETA_EXCLUDED = ['sub-007', 'sub-009']
HARVARD_ATLAS_GA = [i for i in range(21, 39)]
CHINESE_ATLAS_GA = [i for i in range(22, 36)]
SB_ATLAS_GA = [i for i in range(21, 35)] + [25]


def get_ga(condition, center_group):
    ga_list = []

    # Add atlases' GAs
    if center_group == 'Atlases':
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
                    center = 'FeTA'
                elif tsv == INFO_DATA_TSV2:
                    if line[4] == '':
                        center = 'UHL'
                    elif line[4] == 'UCLH':
                        center = 'UCLH'
                    else:
                        center = 'UK'
                elif tsv == INFO_DATA_TSV3:
                    if 'PT' in pat_id:
                        center = 'MUV'
                    else:
                        center = 'KCL'
                else:
                    center = 'UHL'
                if cond == condition and center == center_group:
                    ga_list.append(ga)
    print('\n%s - %s' % (condition, center_group))
    print('%d cases' % len(ga_list))
    return np.array(ga_list)


def main():
    sns.set(font_scale=SNS_FONT_SCALE)
    sns.set_style("whitegrid")
    ncols = len(CONDITIONS)
    nrows = len(SPLIT_CENTERS)
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(BOXPLOT_SIZE[0] * ncols, BOXPLOT_SIZE[1] * nrows),
    )
    for j, condition in enumerate(CONDITIONS):
        for i, center_type in enumerate(SPLIT_CENTERS):
            ga = get_ga(condition, center_type)
            if ga.size > 0:
                g = sns.distplot(
                    ga,
                    kde=False,
                    rug=False,
                    norm_hist=False,
                    bins=range(19, 40),
                    hist_kws=dict(edgecolor="black", linewidth=3),
                    color='darkgreen' if center_type in ['FeTA', 'Atlases'] else 'royalblue',
                    ax=ax[i,j],
                )

            if ga.size > 0:
                ax[i,j].set_title(
                    'Total: %d 3D MRIs' % ga.size,
                    fontsize=FONT_SIZE_NB_CASES,
                )

            # X axis
            ax[i,j].set_xlim((19, 40))
            ax[i,j].set(xticks=[20 + i for i in range(0, 21, 4)])
            if j == 0:
                column_name = CENTER_TO_DISPLAY[center_type]
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

    # Adjust the margins between the subplots
    fig.subplots_adjust(wspace=0.08, hspace=0.15)

    # Remove extra empty space
    fig.tight_layout()

    # Save the figure
    save_name = 'ga_histograms_thesis.pdf'
    save_path = os.path.join(OUTPUT_PATH, save_name)
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    fig.savefig(save_path, bbox_inches='tight')
    print('Figure saved in', save_path)


if __name__ == '__main__':
    print('\033[93mMake Dataset Figure with the Distribution of Gestational Ages for each Group.\033[0m')
    main()
