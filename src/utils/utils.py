import csv
from src.utils.definitions import *


def get_feta_info():
    patid_to_ga = {}
    patid_to_cond = {}
    for tsv in [INFO_DATA_TSV, INFO_DATA_TSV2, INFO_TRAINING_DATA_TSV]:
        first_line = True
        with open(tsv) as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                if first_line:
                    first_line = False
                    continue
                pat_id = line[0]
                cond = line[1]
                ga = int(round(float(line[2])))
                if ga > 38:
                    print('Found ga=%d for %s. Change it to 38 (max value accepted)' % (ga, pat_id))
                    ga = 38
                patid_to_ga[pat_id] = ga
                patid_to_cond[pat_id] = cond
    return patid_to_ga, patid_to_cond
