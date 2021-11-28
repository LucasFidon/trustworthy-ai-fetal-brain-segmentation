import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
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
    for method in METHOD_NAMES:
        m = metrics[method]

        # Set the right list of ROIs
        roi_list = ALL_ROI
        if condition == 'Pathological' and center_type == 'in':
            roi_list = ['white_matter', 'intra_axial_csf', 'cerebellum']
        elif center_type == 'out':
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
                line_base = [i, method]
                metric_name = '%s_%s' % (metric, roi)
                line = line_base + [roi, m[metric_name][i]]
                raw_data.append(line)

    df = pd.DataFrame(raw_data, columns=columns)
    return df

def main(metric_name, condition, center_type):
    df = create_df(metric_name, condition, center_type)

    # for roi in ROI_EVAL:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
    sns.set(style="ticks")
    sns.boxplot(
        data=df,
        hue='Methods',
        y=metric_name,
        x='ROI',
        ax=ax,
        palette='Set3',
    )
    ax.set_title('%s %s-scanner-distibution' % (condition, center_type), fontsize=18, fontweight='bold')
    ax.set_xlabel('Regions of interest', fontsize=16, fontweight='bold')
    ax.set_ylabel('Dice score (in %)', fontsize=16, fontweight='bold')
    sns.move_legend(
        ax,
        "lower left",
        bbox_to_anchor=(1, 0.),
        # title='Species',
    )
    save_name = '%s_%s_%s-distribution.png' % (metric_name, condition.replace(' ', ''), center_type)
    fig.savefig(save_name, bbox_inches='tight')
    print('Figure saved in', save_name)


if __name__ == '__main__':
    # main(METRIC_NAMES[0], CONDITIONS[1], CENTERS[0])
    for metric in METRIC_NAMES:
        for condition in CONDITIONS:
            for center in CENTERS:
                print('\n\033[92m** %s %s %s\033[0m' % (metric, condition, center))
                main(metric, condition, center)
