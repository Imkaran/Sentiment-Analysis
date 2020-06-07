import sklearn.model_selection as model_selection
from sklearn.utils import shuffle
import glob
import os


def read_imdb_data(data_dir='/home/karan/Downloads/aclImdb_v1/data'):
    """Read IMDb movie reviews from given directory.

    Directory structure expected:
    - data/
        - train/
            - pos/
            - neg/
        - test/
            - pos/
            - neg/

    """

    data = []
    labels = []

    # Assume 2 sub-directories: train, test
    for data_type in ['train', 'test']:

        # Assume 2 sub-directories for sentiment (label): pos, neg
        for sentiment in ['pos', 'neg']:

            # Fetch list of files for this sentiment
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)

            # Read reviews data and assign labels
            for f in files:
                with open(f) as review:
                    data.append(review.read())
                    labels.append(sentiment)

            assert len(data) == len(labels), \
                "{}/{} data size does not match labels size".format(data_type, sentiment)

    # Return data, labels as nested dicts
    return data, labels