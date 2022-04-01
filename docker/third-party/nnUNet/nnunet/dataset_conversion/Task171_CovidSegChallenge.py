import os
from scipy.ndimage.measurements import label
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
from lungmask import mask
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data


DATA_FOLDER = '/data/COVID-19-20/COVID-19-20_v2'
TRAIN_DATA_FOLDER = join(DATA_FOLDER, 'Train')
VALID_DATA_FOLDER = join(DATA_FOLDER, 'Validation')
TEST_DATA_FOLDER = '/data/COVID-19-20/COVID-19-20_TestSet'

JUN_DATASET_CT_FOLDER = os.path.join('/data', 'covid_benchmark', 'COVID-19-CT-Seg_20cases')
JUN_DATASET_LESIONS_FOLDER = os.path.join('/data', 'covid_benchmark', 'Infection_Mask')

# Guotai data:
# only binary seg
# non HU intensity
GUOTAI_DATASET_FOLDER = os.path.join('/data', 'UESTC-COVID-19')
GUOTAI_DATASET_PART1 = os.path.join(  # 70 cases labelled by non-experts
    GUOTAI_DATASET_FOLDER,
    'UESTC-COVID-19-20201109T135232Z-001',
    'UESTC-COVID-19',
    'part1',
)
GUOTAI_DATASET_PART2 = os.path.join(  # 50 cases labelled by experts
    GUOTAI_DATASET_FOLDER,
    'UESTC-COVID-19-20201109T135232Z-001',
    'UESTC-COVID-19',
    'part2',
)
GUOTAI_HU_MIN = -1400  # strange value... could it be -1000?
GUOTAI_HU_MAX = 100


# iCovid data
ICOVID_DATASET_FOLDER = os.path.join('/data', 'icovid_raw_data')
LABELS_ICOVID = {
    # Basic lesion classes
    'ggo': 1,
    'consolidation': 2,
    'crazy_paving_pattern': 3,
    'linear_opacity': 2,
    # Super classes
    'combined_pattern': 4,
    'reversed_halo_sign': 4,
    'other_abnormal_tissue': 5,
    'lung': 6,
    'background': 0,
}
PATIENT_ID_TO_EXCLUDE = [
    '1363112652',  # moderate artefact and I can't see some of the lesions segmented
    '1366125607',  # artefact and suspicious seg (completed and reviewed by same person)
    '1413717420',  # strong breathing artefact and suspicious seg
    '1812933091',  # BART: to exclude pat12. nothing seg
    '1868609820',  # BART: to exclude pat13. nothing seg
    '2602703662',  # can't see most of the lesions; noisy seg
    '2762004157',  # mainly other abn and comb pattern; noisy seg
    '2969709397',  # lots of other abn; mix other abn other lesions; can't see some of the lesions
    '3375944345',  # no lesion
    '5925215067',  # not annotated completely (very partial)
    '7414742831',  # can't see the lesions; seg seem noisy
    # '7957238453',  # suspicious: lesion in only one slice
    '8874887577',  # mainly combined pattern; some suspicious seg
]


# PREPROCESSING PARAMS
MIN_HU = -1000  # air
MAX_HU = 100  # max for Guotai's data
MASK_MARGIN = [5, 15, 15]
MIN_NUM_VOXEL_PER_COMP = 100000
LABELS = {
    'lung': 1,
    'lesion': 2,
    'background': 0,
}



def get_patient_name_from_file_name(file_name):
    name = file_name.replace('_ct.nii.gz', '').replace('_seg.nii.gz', '').replace('.nii.gz', '')
    return name

def preprocess(img_path, seg_path=None, mode='challenge'):
    def mask_img(img_np, lung_mask_np):
        x, y, z = np.where(lung_mask_np > 0)
        x_min = max(0, np.min(x) - MASK_MARGIN[0])
        x_max = min(img_np.shape[0], np.max(x) + MASK_MARGIN[0])
        img_np[:x_min, :, :] = 0
        img_np[x_max:, :, :] = 0
        y_min = max(0, np.min(y) - MASK_MARGIN[1])
        y_max = min(img_np.shape[1], np.max(y) + MASK_MARGIN[1])
        img_np[:, :y_min, :] = 0
        img_np[:, y_max:, :] = 0
        z_min = max(0, np.min(z) - MASK_MARGIN[2])
        z_max = min(img_np.shape[2], np.max(z) + MASK_MARGIN[2])
        img_np[:, :, :z_min] = 0
        img_np[:, :, z_max:] = 0
        return img_np

    def postprocess_auto_lung_seg(lung_seg_np):
        # Binarize the lung segmentation
        lung_seg_np[lung_seg_np > 1] = 1
        # Keep only the two largest connected components
        structure = np.ones((3, 3, 3), dtype=np.int)
        labeled, ncomp = label(lung_seg_np, structure)
        size_comp = [
            np.sum(labeled == l) for l in range(1, ncomp + 1)
        ]
        first_largest_comp = np.argmax(size_comp)
        label_first = first_largest_comp + 1
        size_comp[first_largest_comp] = -1
        second_largest_comp = np.argmax(size_comp)
        label_second = second_largest_comp + 1
        # To avoid cases where the two lungs are in the same component
        # and the second largest component is outside the lungs
        # we set a minimum size for the second largest component
        if size_comp[second_largest_comp] < MIN_NUM_VOXEL_PER_COMP:
            label_second = -1
        for i in range(1, ncomp + 1):
            if i in [label_first, label_second]:
                labeled[labeled == i] = 1
            else:
                labeled[labeled == i] = 0
        return labeled

    def update_labels_seg(seg_np, lung_mask_np, mode='normal'):
        new_seg = np.zeros_like(seg_np)
        if mode == 'icovid':
            new_seg[seg_np > 0] = LABELS['lesion']  # all lesion types together
            new_seg[seg_np == LABELS_ICOVID['lung']] = LABELS['lung']
            new_seg[seg_np == LABELS_ICOVID['other_abnormal_tissue']] = LABELS['background']
        else:
            new_seg[lung_mask_np > 0] = LABELS['lung']
            new_seg[seg_np > 0] = LABELS['lesion']
        return new_seg

    def convert_to_sitk(img_np, ref_img_sitk):
        img_sitk = sitk.GetImageFromArray(img_np)
        img_sitk.SetOrigin(ref_img_sitk.GetOrigin())
        img_sitk.SetSpacing(ref_img_sitk.GetSpacing())
        img_sitk.SetDirection(ref_img_sitk.GetDirection())
        return img_sitk

    img = sitk.ReadImage(img_path)
    img_np = sitk.GetArrayFromImage(img)
    if mode == 'guotai':
        # Convert the CT intensities back to HU
        # This has to be done before inference of the lung mask
        img_np = GUOTAI_HU_MIN + (GUOTAI_HU_MAX - GUOTAI_HU_MIN) * img_np
        img = convert_to_sitk(img_np, img)

    # Create the lung mask
    if mode == 'icovid':
        assert seg_path is not None, 'Segmentation is required for iCovid data'
        seg = sitk.ReadImage(seg_path)
        seg_np = sitk.GetArrayFromImage(seg)
        lung_mask_np = np.zeros_like(seg_np)
        lung_mask_np[seg_np > 0] = 1
    else:
        lung_mask_np = mask.apply(img)
        # binarize the mask and keep only the two largest connected components
        lung_mask_np = postprocess_auto_lung_seg(lung_mask_np)

    # Clip the HU intensity
    img_np[img_np < MIN_HU] = MIN_HU
    img_np[img_np > MAX_HU] = MAX_HU

    # Mask the image outside a box containing the lung
    img_np = mask_img(img_np, lung_mask_np)

    # Convert back to SITK image
    img_pre = convert_to_sitk(img_np, img)

    # Seg pre-processing (if available)
    if seg_path is not None:
        seg = sitk.ReadImage(seg_path)
        seg_np = sitk.GetArrayFromImage(seg)
        # Add lung as extra label for the segmentation
        seg_np = update_labels_seg(seg_np, lung_mask_np, mode=mode)
        if mode == 'guotai':
            seg_pre = convert_to_sitk(seg_np, img)  # header is wrong in Guotai's data
        else:
            seg_pre = convert_to_sitk(seg_np, seg)
    else:
        seg_pre = None

    return img_pre, seg_pre



if __name__ == '__main__':
    task_id = 171   # 171: with external data
    task_name = "CovidSegChallenge"

    foldername = "Task%d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagesval = join(out_base, "imagesVal")
    imagestest = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagesval)
    maybe_mkdir_p(imagestest)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    valid_patient_names = []
    test_patient_names = []

    # Training data (Challenge data)
    for f_n in os.listdir(TRAIN_DATA_FOLDER):
        patient_name = get_patient_name_from_file_name(f_n)
        if patient_name in train_patient_names:
            continue
        print('Preprocces', patient_name)
        train_patient_names.append(patient_name)
        img = join(TRAIN_DATA_FOLDER, '%s_ct.nii.gz' % patient_name)
        seg = join(TRAIN_DATA_FOLDER, '%s_seg.nii.gz' % patient_name)
        assert all([
            isfile(img),
            isfile(seg),
        ]), '%s: some files were not found' % patient_name

        img_pre, seg_pre = preprocess(img, seg)
        sitk.WriteImage(img_pre, join(imagestr, patient_name + "_0000.nii.gz"))
        sitk.WriteImage(seg_pre, join(labelstr, patient_name + ".nii.gz"))
    print('Found %d training cases in %s' % (len(train_patient_names), TRAIN_DATA_FOLDER))

    # # Jun dataset
    # jun_patient_names = []
    # for f_n in os.listdir(JUN_DATASET_CT_FOLDER):
    #     if not 'coronacases' in f_n:  # remove data with low quality
    #         continue
    #     patient_name = get_patient_name_from_file_name(f_n)
    #     print('Preprocces', patient_name)
    #     if patient_name in train_patient_names:
    #         continue
    #     jun_patient_names.append(patient_name)
    #     img = join(JUN_DATASET_CT_FOLDER, '%s.nii.gz' % patient_name)
    #     seg = join(JUN_DATASET_LESIONS_FOLDER, '%s.nii.gz' % patient_name)
    #     assert all([
    #         isfile(img),
    #         isfile(seg),
    #     ]), '%s: some files were not found' % patient_name
    #
    #     img_pre, seg_pre = preprocess(img, seg, mode='jun')
    #     sitk.WriteImage(img_pre, join(imagestr, patient_name + "_0000.nii.gz"))
    #     sitk.WriteImage(seg_pre, join(labelstr, patient_name + ".nii.gz"))
    # train_patient_names += jun_patient_names
    # print('Found %d training cases in %s' % (len(jun_patient_names), JUN_DATASET_CT_FOLDER))
    #
    # Guotai data (expert)
    guotai_pat_names = []
    img_folder = os.path.join(GUOTAI_DATASET_PART2, 'image')
    seg_folder = os.path.join(GUOTAI_DATASET_PART2, 'label')
    for f_n in os.listdir(img_folder):
        patient_name = get_patient_name_from_file_name(f_n) + '_part2'
        if patient_name in train_patient_names:
            continue
        print('Preprocces', patient_name)
        guotai_pat_names.append(patient_name)
        img = join(img_folder, f_n)
        seg = join(seg_folder, f_n)
        assert all([
            isfile(img),
            isfile(seg),
        ]), '%s: some files were not found' % patient_name

        img_pre, seg_pre = preprocess(img, seg, mode='guotai')
        sitk.WriteImage(img_pre, join(imagestr, patient_name + "_0000.nii.gz"))
        sitk.WriteImage(seg_pre, join(labelstr, patient_name + ".nii.gz"))
    train_patient_names += guotai_pat_names
    print('Found %d training cases in %s' % (len(guotai_pat_names), GUOTAI_DATASET_PART2))

    # # Guotai data (non-expert)
    # guotai_pat_names = []
    # img_folder = os.path.join(GUOTAI_DATASET_PART1, 'image')
    # seg_folder = os.path.join(GUOTAI_DATASET_PART1, 'label')
    # for f_n in os.listdir(img_folder):
    #     patient_name = get_patient_name_from_file_name(f_n) + '_part1'
    #     if patient_name in train_patient_names:
    #         continue
    #     print('Preprocces', patient_name)
    #     guotai_pat_names.append(patient_name)
    #     img = join(img_folder, f_n)
    #     seg = join(seg_folder, f_n)
    #     assert all([
    #         isfile(img),
    #         isfile(seg),
    #     ]), '%s: some files were not found' % patient_name
    #
    #     img_pre, seg_pre = preprocess(img, seg, mode='guotai')
    #     sitk.WriteImage(img_pre, join(imagestr, patient_name + "_0000.nii.gz"))
    #     sitk.WriteImage(seg_pre, join(labelstr, patient_name + ".nii.gz"))
    # train_patient_names += guotai_pat_names
    # print('Found %d training cases in %s' % (len(guotai_pat_names), GUOTAI_DATASET_PART1))
    #
    # # iCovid data
    # icovid_patient_names = []
    # for f_n in os.listdir(ICOVID_DATASET_FOLDER):
    #     patient_name = f_n
    #     if patient_name in PATIENT_ID_TO_EXCLUDE:
    #         print(patient_name, 'excluded')
    #         continue
    #     print('Preprocces', patient_name)
    #     icovid_patient_names.append(patient_name)
    #     img = join(ICOVID_DATASET_FOLDER, patient_name, 'ct.nii.gz')
    #     seg = join(ICOVID_DATASET_FOLDER, patient_name, 'lesions_seg.nii.gz')
    #     assert all([
    #         isfile(img),
    #         isfile(seg),
    #     ]), '%s: some files were not found' % patient_name
    #
    #     img_pre, seg_pre = preprocess(img, seg, mode='icovid')
    #     sitk.WriteImage(img_pre, join(imagestr, patient_name + "_0000.nii.gz"))
    #     sitk.WriteImage(seg_pre, join(labelstr, patient_name + ".nii.gz"))
    # train_patient_names += icovid_patient_names
    # print('Found %d training cases in %s' % (len(icovid_patient_names), ICOVID_DATASET_FOLDER))
    #
    # print('')
    # print('A total of %s training cases were found' % len(train_patient_names))
    # print('')
    #
    # # Validation data
    # for f_n in os.listdir(VALID_DATA_FOLDER):
    #     patient_name = get_patient_name_from_file_name(f_n)
    #     if patient_name in valid_patient_names:
    #         continue
    #     valid_patient_names.append(patient_name)
    #     img = join(VALID_DATA_FOLDER, '%s_ct.nii.gz' % patient_name)
    #     assert isfile(img), '%s: CT file was not found' % patient_name
    #     img_pre, _ = preprocess(img)
    #     sitk.WriteImage(img_pre, join(imagesval, patient_name + "_0000.nii.gz"))
    # print('Found %d validation cases' % len(valid_patient_names))

    # Test data
    for f_n in os.listdir(TEST_DATA_FOLDER):
        patient_name = get_patient_name_from_file_name(f_n)
        if patient_name in test_patient_names:
            continue
        test_patient_names.append(patient_name)
        img = join(TEST_DATA_FOLDER, '%s_ct.nii.gz' % patient_name)
        assert isfile(img), '%s: CT file was not found' % patient_name
        img_pre, _ = preprocess(img)
        sitk.WriteImage(img_pre, join(imagestest, patient_name + "_0000.nii.gz"))
    print('Found %d testing cases' % len(test_patient_names))

    # Dataset json file
    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "no reference"
    json_dict['licence'] = "no license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "lung",
        "2": "lesion",
    }
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [{
        'image': "./imagesTr/%s.nii.gz" % i,
        "label": "./labelsTr/%s.nii.gz" % i}
        for i in train_patient_names]
    json_dict['test'] = []
    # save_json(json_dict, join(out_base, "dataset.json"))
