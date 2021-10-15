import os


REPO_PATH = '/workspace/trustworthy-ai-fetal-brain-segmentation'


# GENERAL EVALUATION OPTIONS
NUM_CLASS = 9  # number of classes predicted by the model
METHOD_NAMES = ['cnn', 'atlas', 'trustworthy_atlas_only', 'trustworthy']
ALL_ROI = [
    'white_matter', 'intra_axial_csf', 'cerebellum', 'extra_axial_csf',
    'cortical_grey_matter', 'deep_grey_matter', 'brainstem', 'corpus_callosum'
]
CONDITIONS = ['Neurotypical', 'Spina Bifida', 'Pathological']


# EVALUATION
LABELS = {
    'white_matter': [1, 8],  # the cc is part of the wm
    'intra_axial_csf': [2],
    'cerebellum': [3],
    'extra_axial_csf': [4],
    'cortical_grey_matter': [5],
    'deep_grey_matter': [6],
    'brainstem': [7],
    'corpus_callosum': [8],
    'background': [0],
}
METRIC_NAMES = ['dice', 'hausdorff']
MAX_HD = (144. / 2.) * 0.8  # distance from thr center to the border (57.6 mm)


# PARENT FOLDERS
HOME_FOLDER = '/'
WORKSPACE_FOLDER = os.path.join(HOME_FOLDER, 'workspace')
DATA_FOLDER = os.path.join(HOME_FOLDER, 'data')
BASE_FOLDER = os.path.join(DATA_FOLDER, 'Fetal_SRR_and_Seg')
DATA_FOLDER_MICHAEL_GROUP = os.path.join(BASE_FOLDER, 'SRR_and_Seg_Michael_cases_group')
DATA_FOLDER_NADA_GROUP = os.path.join(BASE_FOLDER, 'SRR_and_Seg_Nada_cases_group')


# REGISTRATION
NIFTYREG_PATH = os.path.join(WORKSPACE_FOLDER, 'niftyreg_stable', 'build', 'reg-apps')


# ATLAS FOLDERS
ATLAS_CONTROL_HARVARD = os.path.join(  # GA: 21 -> 37
    DATA_FOLDER,
    'fetal_brain_atlases',
    'Gholipour2017_atlas_NiftyMIC_preprocessed_corrected',
)
ATLAS_CONTROL_CHINESE = os.path.join(  # GA: 22 -> 35
    DATA_FOLDER,
    'fetal_brain_atlases',
    'FBA_Chinese_main_preprocessed_corrected',
)
ATLAS_SB = os.path.join(
    DATA_FOLDER,
    'spina_bifida_atlas',
)


# TESTING DATA
DATA_FOLDER_THOMAS_GROUP1 = os.path.join(  # 23 volumes
    DATA_FOLDER_MICHAEL_GROUP,
    'Abnormal_cases',
)
DATA_FOLDER_THOMAS_GROUP2 = os.path.join(
    DATA_FOLDER_MICHAEL_GROUP,
    'Abnormal_cases_Mar20',
)
CDH_LEUVEN_TESTINGSET = os.path.join(  # 19 CDH cases
    DATA_FOLDER_MICHAEL_GROUP,
    'CDH_Doaa_Aug20',
)
DATA_FOLDER_CONTROLS2_PARTIAL_FULLYSEG = os.path.join(  # 7 controls
    DATA_FOLDER_NADA_GROUP,
    'Controls_2_partial',
)
SB_FRED = os.path.join(  # 46 SB cases
    BASE_FOLDER,
    'SRR_and_Seg_Frederic_cases_group',
    'SB_Fred_corrected_partial',
)
CORRECTED_ZURICH_DATA_DIR = os.path.join(BASE_FOLDER, 'FetalDataZurichCorrected', 'TrainingSet')  # 38 volumes
EXCLUDED_ZURICH_DATA_DIR = os.path.join(BASE_FOLDER, 'FetalDataZurichCorrected', 'TrainingSetExcluded')  # 2 volumes
FETA_IRTK_DIR = os.path.join(DATA_FOLDER, 'FetalDataFeTAChallengeIRTK_Jun21')  # 40 volumes

DATASET_LABELS = {
    DATA_FOLDER_THOMAS_GROUP1:
        ['white_matter', 'intra_axial_csf', 'cerebellum'],
    DATA_FOLDER_THOMAS_GROUP2:
        ['white_matter', 'intra_axial_csf', 'cerebellum'],
    CORRECTED_ZURICH_DATA_DIR:
        ['white_matter', 'intra_axial_csf', 'cerebellum', 'extra_axial_csf', 'cortical_grey_matter', 'deep_grey_matter', 'brainstem'],
    # SURE_EXCLUDED_ZURICH_DATA_DIR:
    #     ['white_matter', 'csf', 'cerebellum', 'external_csf', 'cortical_gm', 'deep_gm', 'brainstem'],
    EXCLUDED_ZURICH_DATA_DIR:
        ['white_matter', 'intra_axial_csf', 'cerebellum', 'extra_axial_csf', 'cortical_grey_matter', 'deep_grey_matter', 'brainstem'],
    FETA_IRTK_DIR:
        ['white_matter', 'intra_axial_csf', 'cerebellum', 'extra_axial_csf', 'cortical_grey_matter', 'deep_grey_matter', 'brainstem'],
    SB_FRED:
        ALL_ROI,
    CDH_LEUVEN_TESTINGSET:
        ['white_matter', 'intra_axial_csf', 'cerebellum', 'extra_axial_csf'],
    DATA_FOLDER_CONTROLS2_PARTIAL_FULLYSEG:
        ALL_ROI,
}


# CSV
INFO_DATA_TSV = os.path.join('/data', 'feta_2.1', 'participants.tsv')
INFO_DATA_TSV2 = os.path.join('/data', 'Fetal_SRR_and_Seg', 'iid_testing_trustworthyai21.tsv')