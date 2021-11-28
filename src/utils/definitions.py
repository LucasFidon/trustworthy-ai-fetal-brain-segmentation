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
CENTERS = ['in', 'out']  # in- or -out of distribution
ATLAS_MARGIN = [2] * 9  # bg, wm, vent, cer, ext-csf, cgm, dgm, bs, cc (in voxels)


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
GRID_SPACING = 4  # in mm (default is 4 mm = 5 voxels x 0.8 mm.voxels**(-1))
BE = 0.1  # NiftyReg default 0.001
LE = 0.3  # NiftyReg default 0.01
LP = 3  # default 3; we do only the lp first level of the pyramid

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
CORRECTED_ZURICH_DATA_DIR = os.path.join(BASE_FOLDER, 'FetalDataZurichCorrected', 'TrainingSet')  # 38 volumes
EXCLUDED_ZURICH_DATA_DIR = os.path.join(BASE_FOLDER, 'FetalDataZurichCorrected', 'TrainingSetExcluded')  # 2 volumes
FETA_IRTK_DIR = os.path.join(DATA_FOLDER, 'FetalDataFeTAChallengeIRTK_Jun21_corrected')  # 40 volumes


DATASET_LABELS = {
    TRAINING_DATA_PREPROCESSED_DIR: ALL_ROI,
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
    SB_FRED: ALL_ROI,
    CDH_LEUVEN_TESTINGSET: ALL_ROI,
    DATA_FOLDER_CONTROLS2_PARTIAL_FULLYSEG: ALL_ROI,
}

# Dictionary that maps dataset path to in- or out- of distribution
DATASET_GROUPS = {
    TRAINING_DATA_PREPROCESSED_DIR: 'in',
    DATA_FOLDER_THOMAS_GROUP1: 'in',
    DATA_FOLDER_THOMAS_GROUP2: 'in',
    CORRECTED_ZURICH_DATA_DIR: 'out',
    # SURE_EXCLUDED_ZURICH_DATA_DIR: 'out,
    EXCLUDED_ZURICH_DATA_DIR: 'out',
    FETA_IRTK_DIR: 'out',
    SB_FRED: 'in',
    CDH_LEUVEN_TESTINGSET: 'in',
    DATA_FOLDER_CONTROLS2_PARTIAL_FULLYSEG: 'in',
}


# CSV
INFO_DATA_TSV = os.path.join('/data', 'feta_2.1', 'participants.tsv')
INFO_DATA_TSV2 = os.path.join('/data', 'Fetal_SRR_and_Seg', 'iid_testing_trustworthyai21.tsv')
INFO_TRAINING_DATA_TSV = os.path.join('/data', 'Fetal_SRR_and_Seg', 'training_trustworthyai21_noatlas.tsv')
