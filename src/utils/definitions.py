import os
import numpy as np


# PATHS OF FOLDERS IN THE REPO
REPO_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_PATH = os.path.join(REPO_PATH, 'output')
REPO_DATA_PATH = os.path.join(REPO_PATH, 'data')


# GENERAL EVALUATION OPTIONS
IMG_RES = 0.8  # in mm; isotropic
NUM_CLASS = 9  # number of classes predicted by the model
METHOD_NAMES = ['cnn', 'atlas', 'trustworthy']
ALL_ROI = [
    'white_matter', 'intra_axial_csf', 'cerebellum', 'extra_axial_csf',
    'cortical_grey_matter', 'deep_grey_matter', 'brainstem', 'corpus_callosum'
]
CONDITIONS = ['Neurotypical', 'Spina Bifida', 'Pathological']
CENTERS = ['in', 'out']  # in- or out-of-distribution


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
MAX_HD = (144. / 2.) * IMG_RES  # distance from thr center to the border (57.6 mm)


# REGISTRATION HYPER-PARAMETERS
GRID_SPACING = 4  # in mm (default is 4 mm = 5 voxels x 0.8 mm.voxels**(-1))
BE = 0.1
LE = 0.3
LP = 3  # default 3; we do only the lp first level of the pyramid
DELTA_GA_CONTROL = 1
DELTA_GA_SPINA_BIFIDA = 3
ATLAS_MARGINS_CONTROL_MM = np.array([1.6, 1.6, 1.1, 1.6, 0.8, 1.4, 2.4, 2.9, 1.1])
ATLAS_MARGINS_CONTROL = ATLAS_MARGINS_CONTROL_MM / IMG_RES
ATLAS_MARGINS_SPINA_BIFIDA_MM = np.array([1.6, 2.0, 1.0, 2.7, 2.3, 2.0, 1.8, 3.4, 1.7])
ATLAS_MARGINS_SPINA_BIFIDA = ATLAS_MARGINS_SPINA_BIFIDA_MM / IMG_RES
MIN_GA = 21
MAX_GA = 38


# PARENT FOLDERS
HOME_FOLDER = '/'
WORKSPACE_FOLDER = os.path.join(HOME_FOLDER, 'workspace')
DATA_FOLDER = os.path.join(HOME_FOLDER, 'data')
BASE_FOLDER = os.path.join(DATA_FOLDER, 'Fetal_SRR_and_Seg')
DATA_FOLDER_MICHAEL_GROUP = os.path.join(BASE_FOLDER, 'SRR_and_Seg_Michael_cases_group')
DATA_FOLDER_NADA_GROUP = os.path.join(BASE_FOLDER, 'SRR_and_Seg_Nada_cases_group')

NIFTYREG_PATH = os.path.join(WORKSPACE_FOLDER, 'third-party', 'niftyreg', 'build', 'reg-apps')


# ATLAS FOLDERS
ATLAS_CONTROL_HARVARD = os.path.join(  # GA: 21 -> 37
    REPO_DATA_PATH,
    'fetal_brain_atlases',
    'Neurotypical_Gholipour2017',
)
ATLAS_CONTROL_CHINESE = os.path.join(  # GA: 22 -> 35
    REPO_DATA_PATH,
    'fetal_brain_atlases',
    'Neurotypical_Wu2021',
)
ATLAS_SB = os.path.join(
    REPO_DATA_PATH,
    'fetal_brain_atlases',
    'SpinaBifida_Fidon2021',
)


# TRAINING DATA
TRAINING_DATA_PREPROCESSED_DIR = os.path.join(
    DATA_FOLDER,
    "fetal_brain_srr_parcellation_Jun21_atlas_autocomplete_partially_sup",
)
CDH_DOAA_DEC19 = os.path.join(
    DATA_FOLDER_MICHAEL_GROUP,
    "CDH_Doaa_Dec19",
)
CONTROLS_DOAA_OCT20 = os.path.join(
    DATA_FOLDER_MICHAEL_GROUP,
    "Controls_Doaa_Oct20_MA",
)
DOAA_BRAIN_LONGITUDINAL_SRR_AND_SEG = os.path.join(
    BASE_FOLDER,
    "Doaa_brain_longitudinal_SRR_and_Seg_MA",
)
LEUVEN_MMC = os.path.join(
    DATA_FOLDER_NADA_GROUP,
    "Leuven_MMC",
)
CDH = os.path.join(
    DATA_FOLDER_NADA_GROUP,
    "CDH",
)
CONTROLS_WITH_EXT_CSF = os.path.join(
    DATA_FOLDER_NADA_GROUP,
    "Controls_with_extcsf_MA",
)

FOLD_0 = [
    "Anon6021676120121019_Study1",
    "Anon6075767420140925_Study1",
    "Anon6094282620150505_Study1",
    "Anon6094282620150612_Study1",
    "Anon6310433520131223_Study1",
    "AutoUploadSubject00021_Study1",
    "AutoUploadSubject00029_Study1",
    "AutoUploadSubject00040_Study1",
    "AutoUploadSubject00049_Study1",
    "AutoUploadSubject00068_Study1",
    "AutoUploadSubject00088_Study1",
    "AutoUploadSubject00093_Study1",
    "AutoUploadSubject00120_Study1",
    "AutoUploadSubject00137_Study1",
    "AutoUploadSubject00139_Study1",
    "AutoUploadSubject00141_Study1",
    "AutoUploadSubject00167_Study1",
    "AutoUploadSubject00215_Study1",
    "UZL00056_Study2",
    "UZL00059_Study1",
    "UZL00059_Study5",
    "UZL00066_Study2",
    "UZL00072_Study2",
    "UZL00085_Study1",
    "UZL00111_Study1",
    "UZL00114_Study1",
    "UZL00131_Study1",
    "UZL10_Study15",
    "UZL10_Study1",
    "UZL7_Study12",
    "UZL7_Study1",
    "UZL9_Study1",
]


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
SB_FRED2 = os.path.join(
    BASE_FOLDER,
    'Fred_additional_cases_Sept2021',
)
CONTROLS_KCL = os.path.join(
    BASE_FOLDER,
    'SRR_and_Seg_KCL',
    'Control'
)
CORRECTED_ZURICH_DATA_DIR = os.path.join(BASE_FOLDER, 'FetalDataZurichCorrected', 'TrainingSet')  # 30 volumes
EXCLUDED_ZURICH_DATA_DIR = os.path.join(BASE_FOLDER, 'FetalDataZurichCorrected', 'TrainingSetExcluded')  # 8 volumes
ZURICH_TEST_DATA_DIR = os.path.join(BASE_FOLDER, 'FetalDataZurichCorrected', 'TestingSet')  # 10 volumes
FETA_IRTK_DIR = os.path.join(DATA_FOLDER, 'FetalDataFeTAChallengeIRTK_Jun21_corrected')  # 40 volumes
SB_VIENNA = os.path.join(DATA_FOLDER_NADA_GROUP, 'vienna_MMC_unoperated')  # 11 cases
UCLH_MMC_2 = os.path.join(DATA_FOLDER_NADA_GROUP, 'UCLH_MMC_2')  # 47 cases
DOAA_BRAIN_LONGITUDINAL_SRR_AND_SEG2 = os.path.join(  # 50 CDH volumes
    BASE_FOLDER,
    "Doaa_brain_longitudinal_SRR_and_Seg_2",
)

DATASET_LABELS = {
    TRAINING_DATA_PREPROCESSED_DIR: ['background'] + ALL_ROI,
    DATA_FOLDER_THOMAS_GROUP1: ['background'] + ALL_ROI,
    DATA_FOLDER_THOMAS_GROUP2: ['background'] + ALL_ROI,
    CORRECTED_ZURICH_DATA_DIR: ['background'] + ALL_ROI,
    EXCLUDED_ZURICH_DATA_DIR: ['background'] + ALL_ROI,
    ZURICH_TEST_DATA_DIR: ['background'] + ALL_ROI,
    FETA_IRTK_DIR: ['background'] + ALL_ROI,
    SB_FRED: ['background'] + ALL_ROI,
    SB_FRED2: ['background'] + ALL_ROI,
    CDH_LEUVEN_TESTINGSET: ['background'] + ALL_ROI,
    DATA_FOLDER_CONTROLS2_PARTIAL_FULLYSEG: ['background'] + ALL_ROI,
    SB_VIENNA: ['background'] + ALL_ROI,
    UCLH_MMC_2: ['background'] + ALL_ROI,
    DOAA_BRAIN_LONGITUDINAL_SRR_AND_SEG2: ['background'] + ALL_ROI,
    CONTROLS_KCL: ['background'] + ALL_ROI,
}


# CSV
INFO_DATA_TSV = os.path.join(REPO_DATA_PATH, 'participants.tsv')  # FeTA challenge data info
INFO_DATA_TSV2 = os.path.join(REPO_DATA_PATH, 'iid_testing_trustworthyai21.tsv')
INFO_DATA_TSV3 = os.path.join(REPO_DATA_PATH, 'ood_testing_trustworthyai21.tsv')
INFO_TRAINING_DATA_TSV = os.path.join(REPO_DATA_PATH, 'training_trustworthyai21_noatlas.tsv')
