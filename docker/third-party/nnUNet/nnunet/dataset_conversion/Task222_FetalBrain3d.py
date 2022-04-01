import shutil
import os
import csv
import numpy as np
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk

DATA_FOLDER = "/data/fetal_brain_srr_parcellation_Oct20_atlas"


def read_data_csv(csv_path):
    t2_path_dict = {}
    seg_path_dict = {}
    with open(csv_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            pat_name = row[0]
            t2_path = row[1]
            seg_path = row[2]
            t2_path_dict[pat_name] = t2_path
            seg_path_dict[pat_name] = seg_path
    return t2_path_dict, seg_path_dict


def main():
    task_name = "Task222_FetalBrain3d"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesVal = join(target_base, "imagesVal")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    patient_names = []

    # Training data
    train_csv = join(DATA_FOLDER, 'training.csv')
    train_t2_paths, train_seg_paths = read_data_csv(train_csv)
    for patient_name in list(train_t2_paths.keys()):
        patient_names.append(patient_name)
        t2 = train_t2_paths[patient_name]
        seg = train_seg_paths[patient_name]

        assert all([
            isfile(t2),
            isfile(seg),
        ]), '%s: some files were not found' % patient_name

        shutil.copy(t2, join(target_imagesTr, patient_name + "_0000.nii.gz"))
        shutil.copy(seg, join(target_labelsTr, patient_name + ".nii.gz"))

    # Validation data -> put in train
    valid_csv = join(DATA_FOLDER, 'validation.csv')
    valid_t2_paths, valid_seg_paths = read_data_csv(valid_csv)
    for patient_name in list(valid_t2_paths.keys()):
        patient_names.append(patient_name)
        t2 = valid_t2_paths[patient_name]
        seg = valid_seg_paths[patient_name]

        assert all([
            isfile(t2),
            isfile(seg),
        ]), '%s: some files were not found' % patient_name

        shutil.copy(t2, join(target_imagesTr, patient_name + "_0000.nii.gz"))
        shutil.copy(seg, join(target_labelsTr, patient_name + ".nii.gz"))

    # Testing data
    test_patient_names = []
    test_csv = join(DATA_FOLDER, 'inference.csv')
    test_t2_paths, test_seg_paths = read_data_csv(test_csv)
    for patient_name in list(test_t2_paths.keys()):
        test_patient_names.append(patient_name)
        t2 = test_t2_paths[patient_name]
        seg = test_seg_paths[patient_name]

        assert all([
            isfile(t2),
            isfile(seg),
        ]), '%s: some files were not found' % patient_name

        shutil.copy(t2, join(target_imagesTs, patient_name + "_0000.nii.gz"))

    json_dict = OrderedDict()
    json_dict['name'] = "FetalBrain3D"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T2",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "white_matter",
        "2": "ventricles",
        "3": "cerebellum",
    }
    json_dict['numTraining'] = len(patient_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_names]
    json_dict['test'] = []

    save_json(json_dict, join(target_base, "dataset.json"))


if __name__ == '__main__':
    main()
