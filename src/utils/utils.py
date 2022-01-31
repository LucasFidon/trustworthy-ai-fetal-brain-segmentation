import csv
from src.utils.definitions import *


def get_feta_info():
    patid_to_ga = {}
    patid_to_cond = {}
    patid_to_center = {}
    patid_to_split = {}
    for tsv in [INFO_DATA_TSV, INFO_DATA_TSV2, INFO_DATA_TSV3, INFO_TRAINING_DATA_TSV]:
        first_line = True
        with open(tsv) as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                if first_line:
                    first_line = False
                    continue
                pat_id = line[0]
                cond = line[1]
                # Get GA
                ga = float(line[2])
                # ga = int(round(float(line[2])))
                # if ga > MAX_GA:
                #     print('Found ga=%d for %s. Change it to %d (max value accepted)' % (ga, pat_id, MAX_GA))
                #     ga = MAX_GA
                # if ga < MIN_GA:
                #     print('Found ga=%d for %s. Change it to %d (min value accepted)' % (ga, pat_id, MIN_GA))
                #     ga = MIN_GA
                # Get center
                if tsv == INFO_DATA_TSV:
                    center = 'out'
                else:
                    center = line[3]
                # Get Training/Test split
                if tsv == INFO_TRAINING_DATA_TSV:
                    split = 'training'
                else:
                    split = 'testing'
                patid_to_ga[pat_id] = ga
                patid_to_cond[pat_id] = cond
                patid_to_center[pat_id] = center
                patid_to_split[pat_id] = split
    return patid_to_ga, patid_to_cond, patid_to_center, patid_to_split
