import csv
from src.utils.definitions import *


class Sample:
    def __init__(self, ga, cond, center, split, srr_quality=None):
        self.ga = ga
        self.cond = cond
        self.center = center
        self.split = split
        self.srr_quality = srr_quality


def round_ga(ga):
    out = int(round(ga))
    if out > MAX_GA:
        # print('Found ga=%d. Change it to %d (max value accepted)' % (out, MAX_GA))
        out = MAX_GA
    if out < MIN_GA:
        # print('Found ga=%d. Change it to %d (min value accepted)' % (out, MIN_GA))
        out = MIN_GA
    return out


def get_feta_info(round_GA=False):
    patid_to_sample = {}
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
                if round_GA:
                    ga = round_ga(ga)
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
                # Get SRR quality
                if tsv == INFO_DATA_TSV:  # FeTA
                    quality = float(line[3])
                else:
                    quality = None
                # Create the sample
                sample = Sample(ga, cond, center, split, quality)
                patid_to_sample[pat_id] = sample
    return patid_to_sample
