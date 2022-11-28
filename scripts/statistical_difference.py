import argparse
import warnings
from scipy.stats import wilcoxon

from make_results_figure import *


parser = argparse.ArgumentParser()
parser.add_argument(
    '--method1',
    type=str,
    help='must be one of %s' % str(list(METHOD_NAME_TO_DISPLAY.keys())),
)
parser.add_argument(
    '--method2',
    type=str,
    help='same options as for --method1. --method1 and --method2 are symmetrical',
)


def is_statistically_different(method1, method2):
    print('*** \033[94mStatistical difference %s vs %s:\033[0m' % (method1, method2))
    for metric_name in METRIC_NAMES:
        print('\n*** \033[93m%s\033[0m' % METRIC_NAMES_TO_DISPLAY[metric_name])
        for i, condition in enumerate(CONDITIONS):
            for j, center_type in enumerate(CENTERS):
                df = create_df(metric_name, condition, center_type, average_roi=True)
                m1 = df[df['Methods'] == METHOD_NAME_TO_DISPLAY[method1]][metric_name].to_numpy()
                m2 = df[df['Methods'] == METHOD_NAME_TO_DISPLAY[method2]][metric_name].to_numpy()
                assert m1.size == m2.size
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, p_value = wilcoxon(m1, m2)
                res = '\033[92mYES\033[0m' if p_value < 0.05 else '\033[91mNO\033[0m'
                print('*** \033[92m%s - %s: %d cases\033[0m' % (condition, center_type, m1.size))
                print('%s p-value=%s' % (res, p_value))
                print('Median: %.2f vs %.2f' % (np.median(m1), np.median(m2)))


if __name__ == '__main__':
    args = parser.parse_args()
    m1 = args.method1
    m2 = args.method2
    is_statistically_different(m1, m2)
