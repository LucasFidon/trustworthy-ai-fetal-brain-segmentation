"""
@brief  This script has been used to select the fetal brain MRIs used for
        the trustworthiness scoring by experts.
        The automatic segmentations are pseudonymised so that
        the raters are blind to the segmentation method they score.
        A csv decode.csv is generated to de-anonymized the segmentations.
        The raters did no have access to this CSV.
        A csv to be filled by the raters is also automatically generated.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
"""

import os
import random
import csv
import nibabel as nib
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.utils import get_feta_info
from src.utils.definitions import *
from run_infer_eval_all import SAVE_FOLDER


SAVE_FOLDER_RANKING = os.path.join(OUTPUT_PATH, 'trustworthiness_scores_fetal_brain')
RANKING_CSV = os.path.join(SAVE_FOLDER_RANKING, 'ranking.csv')
DECODE_CSV = os.path.join(REPO_DATA_PATH, 'decode.csv')


def main():
    patid_sample = get_feta_info()

    # Get all the study folder paths
    studies_id = {
        cond: [] for cond in CONDITIONS
    }
    patid_to_folder = {}
    for data_folder in [FETA_IRTK_DIR, ZURICH_TEST_DATA_DIR, EXCLUDED_ZURICH_DATA_DIR, CORRECTED_ZURICH_DATA_DIR]:
        for f_n in os.listdir(data_folder):
            if '.' in f_n:
                continue
            patid = f_n[:].replace('feta', '')
            sample = patid_sample[patid]
            cond =sample.cond
            study_path = os.path.join(data_folder, f_n)
            patid_to_folder[patid] = study_path
            studies_id[cond].append(patid)

    # Shuffle the studies
    for cond in CONDITIONS:
        random.shuffle(studies_id[cond])

    # Keep only 50 studies
    studies_id['Neurotypical'] = studies_id['Neurotypical'][:20]
    studies_id['Spina Bifida'] = studies_id['Spina Bifida'][:20]
    studies_id['Pathological'] = studies_id['Pathological'][:10]

    # Pseudonymised and copy files
    if not os.path.exists(SAVE_FOLDER_RANKING):
        os.mkdir(SAVE_FOLDER_RANKING)
    rows_ranking = []
    rows_decode = []
    for cond in CONDITIONS:
        for patid in studies_id[cond]:
            print('')
            print('Copy files for %s' % patid)
            folder = patid_to_folder[patid]
            f_n = os.path.split(folder)[1]
            srr = os.path.join(folder, 'srr.nii.gz')
            gt = os.path.join(folder, 'parcellation.nii.gz')
            pred_folder = os.path.join(SAVE_FOLDER, 'nnunet_task225', f_n)
            save_folder = os.path.join(SAVE_FOLDER_RANKING, f_n)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            # Copy the SRR and the manual seg
            save_srr = os.path.join(save_folder, 'srr.nii.gz')
            os.system('rsync -vah --progress %s %s' % (srr, save_srr))
            save_gt = os.path.join(save_folder, 'manual_segmentation.nii.gz')
            os.system('rsync -vah --progress %s %s' % (gt, save_gt))

            # Shuffle the autosegs
            autoseg_names = [
                '%s.nii.gz' % f_n,
                '%s_atlas.nii.gz' % f_n,
                '%s_trustworthy.nii.gz' % f_n,
            ]
            random.shuffle(autoseg_names)
            rows_ranking.append([patid])
            rows_decode.append([patid] + autoseg_names)
            for i, seg in enumerate(autoseg_names):
                pred_seg = os.path.join(pred_folder, seg)
                save_seg = os.path.join(save_folder, 'autoseg_%d.nii.gz' % (i+1))
                # Merge CC to WM
                seg_nii = nib.load(pred_seg)
                seg = seg_nii.get_fdata().astype(np.uint8)
                seg[seg == LABELS['corpus_callosum']] = LABELS['white_matter'][0]
                new_seg_nii = nib.Nifti1Image(seg, seg_nii.affine)
                nib.save(new_seg_nii, save_seg)

    # Save the CSV files
    with open(RANKING_CSV, 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        row1 = ['']
        for c in ALL_ROI:
            if c != 'corpus_callosum':
                row1 += [c] * 3
        writer.writerow(row1)
        row2 = ['Study ID']
        for c in ALL_ROI:
            if c != 'corpus_callosum':
                row2 += ['Auto seg 1', 'Auto seg 2', 'Auto seg 3']
        writer.writerow(row2)
        for row in rows_ranking:
            writer.writerow(row)

    with open(DECODE_CSV, 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Study ID', 'Auto seg 1', 'Auto seg 2', 'Auto seg 3'])
        for row in rows_decode:
            writer.writerow(row)


if __name__ == '__main__':
    random.seed(27)
    main()