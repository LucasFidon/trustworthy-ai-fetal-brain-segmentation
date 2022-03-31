import os
import csv
import openpyxl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import sys
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)
from scripts.create_scoring_files import DECODE_CSV
from src.utils.utils import get_feta_info
from src.utils.definitions import *

RANKING_CSV = {
    'Leuven': os.path.join(REPO_DATA_PATH, 'ranking_MA.xlsx'),
    'Vienna': os.path.join(REPO_DATA_PATH, 'ranking_Vienna.xlsx'),
    'Zurich_AJ': os.path.join(REPO_DATA_PATH, 'aj_ranking.xlsx'),
    'Zurich_AB': os.path.join(REPO_DATA_PATH, 'A_BINK_ranking.xlsx'),
}

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
    patid_sample = get_feta_info()
    patid_to_decode =  read_decode_csv()
    columns = ['Rating Center', 'Study', 'Condition', 'ROI', 'Methods', 'Scores', 'SRR Quality Scores']
    raw_data = []

    for center in list(RANKING_CSV.keys()):
        ranking_csv = RANKING_CSV[center]
        obj = openpyxl.load_workbook(ranking_csv)
        sheet = obj.active
        is_first = True
        is_second = False
        class_row = None
        method_row = None
        for row in sheet.iter_rows(values_only=True):
            if is_first:
                is_first = False
                is_second = True
                class_row = row[1:]
                continue
            elif is_second:
                is_second = False
                method_n = row[1:]
                method_row = [int(n[-1]) - 1 for n in method_n]
                continue
            patid = row[0]
            sample = patid_sample[patid]
            scores = row[1:]

            for i in range(len(class_row)):  # num_class x num_method
                method_num = method_row[i]
                roi = class_row[i]
                segmentation = patid_to_decode[patid][method_num]
                if 'atlas' in segmentation:
                    method = 'Fallback'
                elif 'trust' in segmentation:
                    method ='TW-AI'
                else:
                    method = 'AI'
                r = scores[i]
                line = [center, patid, sample.cond, ROI_NAMES_TO_DISPLAY[roi], method, r, sample.srr_quality]
                raw_data.append(line)

    df = pd.DataFrame(raw_data, columns=columns)
    return df


def main(center='all'):
    df = read_ranking()
    if center != 'all':
        # Keep only the results from the rating center 'center'
        df = df[df['Rating Center'] == center]
    sns.set(font_scale=SNS_FONT_SCALE+1.5)
    sns.set_style("whitegrid")
    nrows = len(CONDITIONS)
    ncols = 1
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(BOXPLOT_SIZE[0] * ncols, BOXPLOT_SIZE[1] * nrows),
    )
    num_methods = 3
    num_ROIs = 7
    for i, condition in enumerate(CONDITIONS):
        data = df[df['Condition'] == condition]
        num_cases = len(data) / (num_methods * num_ROIs)
        print('Found scores for %d %s cases' % (num_cases, condition))
        g = sns.boxplot(
                data=data,
                hue='Methods',
                y='Scores',
                x='ROI',
                ax=ax[i],
                palette='colorblind',
                fliersize=10,
                linewidth=3,
                hue_order=['AI', 'Fallback', 'TW-AI'],
                order=[ROI_NAMES_TO_DISPLAY[roi] for roi in ALL_ROI[:-1]],
            )
        if i < len(CONDITIONS) - 1:
            ax[i].set(xlabel=None)
        else:
            ax[i].set_xlabel(
                '\nROI' ,
                fontsize=FONT_SIZE_AXIS,
                fontweight='bold',
            )
        ax[i].set_ylabel(
                CONDITION_NAMES_TO_DISPLAY[condition] + '\n' ,
                fontsize=FONT_SIZE_AXIS,
                fontweight='bold',
            )
        ax[i].set(ylim=YAXIS_LIM)
        g.set(yticks=YTICKS)
        sns.move_legend(ax[i], 'lower left', bbox_to_anchor=(1, 0.))

    fig.suptitle(
        'Trustworthiness scores per ROI for out-of-distribution 3D MRIs',
        fontsize=55,
    )
    # Remove extra empty space
    fig.tight_layout()
    # Save the figure
    if center != 'all':
        save_name = 'scores_%s.pdf' % center
    else:
        save_name = 'scores.pdf'
    save_path = os.path.join(OUTPUT_PATH, save_name)
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    fig.savefig(save_path, bbox_inches='tight')
    print('Figure saved in', save_path)


def main_aggregated(center='all'):
    df = read_ranking()
    if center != 'all':
        # Keep only the results from the rating center 'center'
        df = df[df['Rating Center'] == center]
        # Average scores across ROIs
        df = df.groupby(['Study', 'Condition', 'Methods'])['Scores'].mean().reset_index()
    else:
        # Average scores across ROIs
        df = df.groupby(['Rating Center', 'Study', 'Condition', 'Methods'])['Scores'].mean().reset_index()

    sns.set(font_scale=SNS_FONT_SCALE+1.6)
    sns.set_style("whitegrid")
    nrows = 1
    ncols = len(CONDITIONS)
    # Fig aggregated
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12 * ncols, 12 * nrows),
    )
    for i, condition in enumerate(CONDITIONS):
        data = df[df['Condition'] == condition]
        g = sns.boxplot(
                data=data,
                x='Methods',
                y='Scores',
                ax=ax[i],
                palette='colorblind',
                fliersize=10,
                linewidth=3,
                order=['AI', 'Fallback', 'TW-AI'],
            )
        ax[i].set_xlabel(
                '\n' + CONDITION_NAMES_TO_DISPLAY[condition],
                fontsize=50,
                fontweight='bold',
            )
        ax[i].set(ylim=YAXIS_LIM)
        g.set(yticks=YTICKS)
        if i == 0:
            ax[i].set_ylabel(
                'Mean-ROI score' + '\n',
                fontsize=50,
                fontweight='bold',
            )
        else:
            ax[i].set(ylabel=None)
    fig.suptitle(
        'Mean-ROI trustworthiness score for out-of-distribution 3D MRIs',
        fontsize=65,
    )
    # Remove extra empty space
    fig.tight_layout()
    # Save the figure
    if center != 'all':
        save_name = 'scores_aggregated_%s.pdf' % center
    else:
        save_name = 'scores_aggregated.pdf'
    save_path = os.path.join(OUTPUT_PATH, save_name)
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    fig.savefig(save_path, bbox_inches='tight')
    print('Figure saved in', save_path)


def main_aggregated_inter_rater_std():
    df = read_ranking()
    # Average scores across ROIs
    df = df.groupby(['Rating Center', 'Study', 'Condition', 'Methods'])['Scores'].mean().reset_index()
    # Compute the std across raters
    df = df.groupby(['Study', 'Condition', 'Methods'])['Scores'].std().reset_index()

    sns.set(font_scale=SNS_FONT_SCALE+1.6)
    sns.set_style("whitegrid")
    nrows = 1
    ncols = len(CONDITIONS)
    # Fig aggregated
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12 * ncols, 12 * nrows),
    )
    for i, condition in enumerate(CONDITIONS):
        data = df[df['Condition'] == condition]
        g = sns.boxplot(
                data=data,
                x='Methods',
                y='Scores',
                ax=ax[i],
                palette='colorblind',
                fliersize=10,
                linewidth=3,
                order=['AI', 'Fallback', 'TW-AI'],
            )
        ax[i].set_xlabel(
                '\n' + CONDITION_NAMES_TO_DISPLAY[condition],
                fontsize=50,
                fontweight='bold',
            )
        if i == 0:
            ax[i].set_ylabel(
                'Mean-ROI score' + '\n',
                fontsize=50,
                fontweight='bold',
            )
        else:
            ax[i].set(ylabel=None)
    fig.suptitle(
        'Inter-rater Std of mean-ROI trustworthiness scores',
        fontsize=65,
    )
    # Remove extra empty space
    fig.tight_layout()
    # Save the figure
    save_name = 'scores_aggregated_inter_rater_std.pdf'
    save_path = os.path.join(OUTPUT_PATH, save_name)
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    fig.savefig(save_path, bbox_inches='tight')
    print('Figure saved in', save_path)


def main_scores_vs_quality():
    df = read_ranking()
    # Average scores across ROIs
    df = df.groupby(['Rating Center', 'Study', 'Condition', 'Methods', 'SRR Quality Scores'])['Scores'].mean().reset_index()

    sns.set(font_scale=SNS_FONT_SCALE+1.6)
    sns.set_style("whitegrid")
    nrows = 1
    ncols = len(CONDITIONS)
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(24 * ncols, 24 * nrows),
    )
    for i, condition in enumerate(CONDITIONS):
        data = df[df['Condition'] == condition]
        r = data['Scores'].corr(data['SRR Quality Scores'])
        print('Pearson r=%f' % r)
        g = sns.regplot(
            data=data,
            x="SRR Quality Scores",
            y="Scores",
            ax=ax[i],
        )
        ax[i].set_title(
            CONDITION_NAMES_TO_DISPLAY[condition] + '\n',
            fontsize=70,
            fontweight='bold',
        )
        ax[i].set_xlabel(
            '3D MRI Quality Score',
            fontsize=50,
            fontweight='bold',
        )
        ax[i].set_ylabel(
            'Trustworthiness Score',
            fontsize=50,
            fontweight='bold',
        )
    fig.suptitle(
        'Trustworthiness scores vs 3D MRI quality scores',
        fontsize=85,
    )
    # Remove extra empty space
    fig.tight_layout()
    # Save the figure
    save_name = 'scores_aggregated_vs_srr_quality.pdf'
    save_path = os.path.join(OUTPUT_PATH, save_name)
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    fig.savefig(save_path, bbox_inches='tight')
    print('Figure saved in', save_path)


if __name__ == '__main__':
    print('\033[93mMake figures for the analysis of the expert scores\033[0m')
    main()
    main_aggregated()
    main_scores_vs_quality()
    main_aggregated_inter_rater_std()
    for center in list(RANKING_CSV.keys()):
        main(center)
        main_aggregated(center)
