import os
import numpy as np
from collections import OrderedDict
import SimpleITK as sitk
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data


def get_identifier(img_path):
    group_id = os.path.split(os.path.split(img_path)[0])[1]
    id = os.path.split(img_path)[1].split('.')[0]
    identifier = '%s_%s' % (group_id, id)
    return identifier


def load_id_list(txt_file_path):
    with open(txt_file_path, 'r') as f:
        id_list = f.read()
    return id_list


def unstack(img_path, seg_path):
    # Load image and segmention as numpy arrays
    img = sitk.ReadImage(img_path)
    img_npy = sitk.GetArrayFromImage(img)
    seg = sitk.ReadImage(seg_path)
    seg_npy = sitk.GetArrayFromImage(seg)
    # Unstack the slices
    n_stack = np.min(img.shape)
    img_slices = np.dsplit(
        img.transpose(np.argsort(img_npy.shape)[::-1]), n_stack)
    seg_slices = np.dsplit(
        seg.transpose(np.argsort(seg_npy.shape)[::-1]), n_stack)
    return img_slices, seg_slices


if __name__ == '__main__':
    task_name = "Task111_FetalBrain2d"
    # For JADE
    raw_data_folder = '/home_directory/data/FetalBrainSegmentation_Dataset'

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesVal = join(target_base, "imagesVal")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    groups = {
        'GroupA': {'val': 'list_validation_h_files.txt',
                   'test': 'list_inference_h_files.txt'},
        'GroupB1': {'val': 'list_validation_p_files.txt',
                   'test': 'list_inference_p1_files.txt'},
        'GroupB2': {'val': 'list_validation_p_files.txt',
                   'test': 'list_inference_p2_files.txt'},
        'GroupC': {'val': 'list_validation_p_files.txt',
                   'test': 'list_inference_C_files.txt'},
        'GroupD': {'val': 'list_validation_p_files.txt',
                   'test': 'list_inference_D_files.txt'},
        'GroupE': {'val': 'list_validation_p_files.txt',
                   'test': 'list_inference_E_files.txt'},
        'GroupF': {'val': 'list_validation_p_files.txt',
                   'test': 'list_inference_F_files.txt'},
    }

    train_slices_names = []
    valid_cases_names = []
    test_cases_names = []

    for group in list(groups.keys()):
        group_path = os.path.join(raw_data_folder, group)
        img_name_list = [
            n for n in os.listdir(group_path)
            if not n.startswith('.') and 'Image' in n
        ]
        val_id_list = load_id_list(groups[group]['val'])
        test_id_list = load_id_list(groups[group]['test'])
        for image_name in img_name_list:
            identifier = '%s_%s' % (group, image_name.split('.')[0])
            img_path = os.path.join(group_path, image_name)
            seg_path = img_path.replace('Image', 'Label')
            pat_id = image_name.replace('_Image.nii.gz', '')
            # Validation data (full volume)
            if pat_id in val_id_list:
                shutil.copy(img_path, join(target_imagesVal, identifier + "_0000.nii.gz"))
                valid_cases_names.append(identifier)
            # Testing data (full volume)
            elif pat_id in test_id_list:
                shutil.copy(img_path, join(target_imagesTs, identifier + "_0000.nii.gz"))
                test_cases_names.append(identifier)
            # Training data (slices)
            else:
                img_slices, seg_slices = unstack(img_path, seg_path)
                for i in range(len(img_slices)):
                    img_slice = sitk.GetImageFromArray(img_slices[i])
                    seg_slice = sitk.GetImageFromArray(seg_slices[i])
                    slice_name = '%s_%s' % (identifier, str(i).zfill(3))
                    import pdb
                    pdb.set_trace()
                    sitk.WriteImage(
                        img_slice,
                        os.path.join(target_imagesTr, '%s_0000.nii.gz' % slice_name),
                    )
                    sitk.WriteImage(
                        seg_slice,
                        os.path.join(target_labelsTr, '%s.nii.gz' % slice_name),
                    )
                    train_slices_names.append(slice_name)
        print('Found %d training slices.' % len(train_slices_names))


    # Dataset json file
    json_dict = OrderedDict()
    json_dict['name'] = "FetalBrain2d"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "no reference"
    json_dict['licence'] = "no license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T2",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "brain",
    }
    json_dict['numTraining'] = len(train_slices_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             train_slices_names]
    json_dict['test'] = []
    save_json(json_dict, join(target_base, "dataset.json"))
