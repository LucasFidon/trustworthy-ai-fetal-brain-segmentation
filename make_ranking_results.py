import os
import csv
import openpyxl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from create_ranking_files import get_feta_info, DECODE_CSV
from src.utils.definitions import *

RANKING_CSV = '/data/ranking_fetal_brain/ranking_v0.xlsx'

BOXPLOT_SIZE = [25, 10]  # Size of each subplot
FONT_SIZE_AXIS = 55
SNS_FONT_SCALE = 2.8
CONDITION_NAMES_TO_DISPLAY = {
    'Neurotypical': 'Neurotypical',
    'Spina Bifida': 'Spina Bifida',
    'Pathological': 'Other Pathologies',
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
YAXIS_LIM = (-0.1, 5.1)
YTICKS = [i for i in range(6)]


def read_decode_csv():
    patid_to_decode = {}
    with open(DECODE_CSV, newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        is_first = True
        for row in reader:
            if is_first:
                is_first = False
                continue
            patid_to_decode[row[0]] = row[1:]
    return patid_to_decode

def read_ranking():
    _, patid_to_cond, _, _ = get_feta_info()
    patid_to_decode =  read_decode_csv()
    columns = ['Condition', 'ROI', 'Methods', 'Scores']
    raw_data = []

    obj = openpyxl.load_workbook(RANKING_CSV)
    sheet = obj.active
    is_first = True
    is_second = False
    class_row = None
    for row in sheet.iter_rows(values_only=True):
        if is_first:
            is_first = False
            is_second = True
            class_row = row[1:]
            continue
        elif is_second:
            is_second = False
            continue
        patid = row[0]
        cond = patid_to_cond[patid]
        scores = row[1:]
        i = 0

        for method_num in range(3):
            segmentation = patid_to_decode[patid][method_num]
            if 'atlas' in segmentation:
                method = 'Fallback'
            elif 'trust' in segmentation:
                method ='Trustworthy AI'
            else:
                method = 'AI'
            for _ in range(7):
                roi = class_row[i]
                r = scores[i]
                if r is not None:
                    line = [cond, ROI_NAMES_TO_DISPLAY[roi], method, r]
                    raw_data.append(line)
                i += 1
    df = pd.DataFrame(raw_data, columns=columns)
    return df


def main():
    df = read_ranking()
    sns.set(font_scale=SNS_FONT_SCALE)
    sns.set_style("whitegrid")
    nrows = len(CONDITIONS)
    ncols = 1
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(BOXPLOT_SIZE[0] * ncols, BOXPLOT_SIZE[1] * nrows),
    )
    for i, condition in enumerate(CONDITIONS):
        data = df[df['Condition'] == condition]
        g = sns.boxplot(
                data=data,
                hue='Methods',
                y='Scores',
                x='ROI',
                ax=ax[i],
                palette='colorblind',
                fliersize=10,
                linewidth=3,
                hue_order=['AI', 'Fallback', 'Trustworthy AI'],
                order=[ROI_NAMES_TO_DISPLAY[roi] for roi in ALL_ROI[:-1]],
            )
        ax[i].set_ylabel(
                CONDITION_NAMES_TO_DISPLAY[condition] + '\n' ,
                fontsize=FONT_SIZE_AXIS,
                fontweight='bold',
            )
        ax[i].set(ylim=YAXIS_LIM)
        g.set(yticks=YTICKS)
        sns.move_legend(ax[i], 'lower left', bbox_to_anchor=(1, 0.))
    # Remove extra empty space
    fig.tight_layout()
    # Save the figure
    save_name = 'scores.png'
    fig.savefig(save_name, bbox_inches='tight')
    print('Figure saved in', save_name)


if __name__ == '__main__':
    main()
