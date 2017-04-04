import argparse
import errno
import logging
import tempfile

from ia_kdsb17.image_prep.helpers import patient_average

logger = logging.getLogger(__name__)


def is_writeable(path):
    try:
        testfile = tempfile.TemporaryFile(dir=path)
        testfile.close()
    except OSError as e:
        if e.errno == errno.EACCES:  # 13
            return False
        e.filename = path
        raise
    return True


def main():
    parser = argparse.ArgumentParser(description='Simple 2D CNN for Kaggle Data Science Bowl 2017')
    parser.add_argument("data_dir", help="The directory containing the patient image files")
    parser.add_argument("labels_file", help="The csv file containing the labels for the data instances")
    parser.add_argument("results_dir", help="The directory to write the results to")
    args = parser.parse_args()

    if not is_writeable(args.results_dir):
        logger.error('results_dir {} is not writeable'.format(args.results_dir))
        return

    averaged_image_genetator = patient_average(args.data_dir)



if __name__ == "__main__":
    main()