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
    'in': 'In-scanner Distribution',
    'out': 'Out-of-scanner Distribution',
}
COLOR = {
    'training': 'green',
    'testing': 'blue',
}
BOXPLOT_SIZE = [15, 10]  # Size of each subplot
FONT_SIZE_AXIS = 55
FONT_SIZE_NB_CASES = 45
SNS_FONT_SCALE = 2.8

def get_ga(condition, center_type, data_split):
    ga_list = []
    for tsv in [INFO_DATA_TSV, INFO_DATA_TSV2, INFO_DATA_TSV3, INFO_TRAINING_DATA_TSV]:
        first_line = True
        with open(tsv) as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                if first_line:
                    first_line = False
                    continue
                pat_id = line[0]
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
    for i, condition in enumerate(CONDITIONS):
        for j, split_center in enumerate(SPLIT_CENTERS):
            if split_center == 'training':
                split = 'training'
                center_type = 'in'
            else:
                split, center_type = split_center.split('_')
            ga = get_ga(condition, center_type, split)
            if ga.size > 0:
                sns.histplot(
                    data=ga,
                    stat='count',
                    fill=True,
                    ax=ax[i,j],
                    color=COLOR[split],
                    binwidth=1,
                    binrange=(19, 40),
                )

            if ga.size > 0:
                ax[i,j].set_title(
                    'Total: %d cases' % ga.size,
                    fontsize=FONT_SIZE_NB_CASES,
                )

            # X axis
            ax[i,j].set_xlim((19, 40))
            if i == nrows - 1:
                column_name ='\n'
                if split == 'training':
                    column_name += 'Training\n'
                    column_name += CENTER_TYPES_TO_DISPLAY[center_type]
                else:
                    column_name += 'Testing\n'
                    column_name += CENTER_TYPES_TO_DISPLAY[center_type]
                ax[i,j].set_xlabel(
                    column_name,
                    fontsize=FONT_SIZE_AXIS,
                    fontweight='bold',
                )
            else:
                ax[i,j].set(xlabel=None)

            # Y axis
            if j == 0:
                ax[i,j].set_ylabel(
                    CONDITION_NAMES_TO_DISPLAY[condition] + '\n' ,
                    fontsize=FONT_SIZE_AXIS,
                    fontweight='bold',
                )
            else:
                ax[i,j].set(ylabel=None)
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
