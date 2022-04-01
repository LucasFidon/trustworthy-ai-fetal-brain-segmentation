import shutil
from copy import deepcopy

from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from multiprocessing import Pool
from deep_gmm.qda import QDA
from nnunet.postprocessing.connected_components import apply_postprocessing_to_folder, load_postprocessing


# def merge_files(args):
def merge_files(gmm_path, files, properties_file, out_file, only_keep_largest_connected_component,
                min_region_size_per_class, override, store_npz):
    # gmm_path, files, properties_file, out_file, only_keep_largest_connected_component, min_region_size_per_class, override, store_npz = args
    # Load the GMM model
    gmm = QDA()
    gmm.load(gmm_path)
    if override or not isfile(out_file):
        embedding = [np.load(f) for f in files]
        embedding = np.concatenate(embedding, axis=0)
        square_mahalanobis = gmm.mahalanobis_distance(embedding)
        log_proba = (-1.) * square_mahalanobis  # log class probability map up to an additive constant
        # N.B.: this is only because covariances are tied and the class prior is uniform.
        # log_proba_pred = gmm.inference(embedding)  # used in place of the proba because only the argmax matters
        props = load_pickle(properties_file)
        save_segmentation_nifti_from_softmax(log_proba, out_file, props, 3, None, None, None, force_separate_z=None)
        if store_npz:
            print('Save the squared Mahalanobis distance map in ', out_file[:-7] + ".npy")
            np.save(out_file[:-7] + ".npy", square_mahalanobis)
            save_pickle(props, out_file[:-7] + ".pkl")


def merge(gmm_path, folders, output_folder, threads, override=True, postprocessing_file=None, store_npz=False):
    maybe_mkdir_p(output_folder)

    if postprocessing_file is not None:
        output_folder_orig = deepcopy(output_folder)
        output_folder = join(output_folder, 'not_postprocessed')
        maybe_mkdir_p(output_folder)
    else:
        output_folder_orig = None

    patient_ids = [subfiles(i, suffix="_embedding.npy", join=False) for i in folders]
    patient_ids = [i for j in patient_ids for i in j]
    patient_ids = [i[:-14] for i in patient_ids]
    patient_ids = np.unique(patient_ids)

    for f in folders:
        assert all([isfile(join(f, i + "_embedding.npy")) for i in patient_ids]), "Not all patient npy are available in " \
                                                                        "all folders"
        assert all([isfile(join(f, i + "_embedding.pkl")) for i in patient_ids]), "Not all patient pkl are available in " \
                                                                        "all folders"

    plans = load_pickle(join(folders[0], "plans.pkl"))
    only_keep_largest_connected_component, min_region_size_per_class = plans['keep_only_largest_region'], \
                                                                       plans['min_region_size_per_class']

    # files = []
    # property_files = []
    # out_files = []
    for p in patient_ids:
        emb_list = [join(f, p + "_embedding.npy") for f in folders]
        prop = join(folders[0], p + "_embedding.pkl")
        out_file = join(output_folder, p + ".nii.gz")
        merge_files(gmm_path, emb_list, prop, out_file, only_keep_largest_connected_component,
                    min_region_size_per_class, override, store_npz)
        # files.append([join(f, p + "_embedding.npy") for f in folders])
        # property_files.append(join(folders[0], p + "_embedding.pkl"))
        # out_files.append(join(output_folder, p + ".nii.gz"))

    # p = Pool(threads)
    # p.map(merge_files, zip([gmm_path] * len(out_files), files, property_files, out_files, [only_keep_largest_connected_component] * len(out_files),
    #                        [min_region_size_per_class] * len(out_files), [override] * len(out_files), [store_npz] * len(out_files)))
    # p.close()
    # p.join()
    #
    # if postprocessing_file is not None:
    #     for_which_classes, min_valid_obj_size = load_postprocessing(postprocessing_file)
    #     print('Postprocessing...')
    #     apply_postprocessing_to_folder(output_folder, output_folder_orig,
    #                                    for_which_classes, min_valid_obj_size, threads)
    #     shutil.copy(postprocessing_file, output_folder_orig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="This script will merge embedding and compute segmentation predictions "
                                                 "using the gmm model given with --gmm."
                                                 " The embedding must have been prdicted with the "
                                                 "--save_npz_embedding option!"
                                                 "You need to specify a postprocessing file so that "
                                                 "we know here what postprocessing must be applied. Failing to do so "
                                                 "will disable postprocessing")
    parser.add_argument('-g', '--gmm', help='path to the gmm model parameters to use.')
    parser.add_argument('-f', '--folders', nargs='+',
                        help="list of embedding folders to merge generated using the --save_npz_embedding option. "
                             "All folders must contain npy files", required=True)
    parser.add_argument('-o', '--output_folder', help="where to save the results", required=True, type=str)
    parser.add_argument('-t', '--threads', help="number of threads used to saving niftis", required=False, default=2,
                        type=int)
    parser.add_argument('-pp', '--postprocessing_file', help="path to the file where the postprocessing configuration "
                                                             "is stored. If this is not provided then no postprocessing "
                                                             "will be made. It is strongly recommended to provide the "
                                                             "postprocessing file!",
                        required=False, type=str, default=None)
    parser.add_argument('--npz', action="store_true", required=False,
                        help="stores npz of the squared Mahalanobis distance map and properties pkl")

    args = parser.parse_args()

    gmm_path = args.gmm
    folders = args.folders
    threads = args.threads
    output_folder = args.output_folder
    pp_file = args.postprocessing_file
    npz = args.npz

    merge(gmm_path, folders, output_folder, threads, override=True, postprocessing_file=pp_file, store_npz=npz)


if __name__ == "__main__":
    main()
