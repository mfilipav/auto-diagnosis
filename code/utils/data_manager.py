import os
import sys

sys.path.insert(0, "../")

from collections import Counter
from constants import Keys, Tables, DATA_DIR
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np
import logging
from utils.icd9 import ICD9
import networkx as nx
from itertools import product


class Dataset:
    """
    Creates a dataset for MIMIC-III database discharge summaries with
    corresponding diagnosis codes. Performs train-test split with 90-10
    percentage. Persists the processed dataset as pkl file.

    Parameters
    ----------
    nrows : int
        Number of discharge records to be read from MIMIC-III notes. If None
        all records are considered.

    augment_labels : bool
        Set it to True if label list for each document should be augmented by
        the parents of each label. ICD9 code hierarchy provided by this repo
        https://github.com/sirrice/icd9

    top_n : int
        If provided, filters out labels such that only top n labels in terms
        of frequency remains

    Attributes
    ----------
    X : list
        List of medical discharge records from MIMIC-III database
    X_train: list
        Training split of medical discharge records from MIMIC-III database
    X_test: list
        Test split of medical discharge records from MIMIC-III database
    y_raw: list
        List of lists of ICD9 diagnosis codes corresponding to each document
    y_train_raw: list
        Train split of list of lists of ICD9 diagnosis codes corresponding
        to each document
    y_test_raw: list
        Test split of list of lists of ICD9 diagnosis codes corresponding
        to each document
    y_train: ndarray
        Hot-encoded version of y_train_raw
    y_test: ndarray
        Hot-encoded version of y_train_raw
    binarizer: sklearn.preprocessing.MultiLabelBinarizer
        binarizer which is used to transform the label lists into hot encoded
        versions.
    labels: list
        list of labels where each entry corresponds to the index in the
        y_train or y_test

    Notes
    -----

    If the test split includes ICD codes that are not included in the training
    set, those labels are automatically removed from the test set since there
    is no sample to support inference.

    """

    def __init__(self, nrows=None, augment_labels=False, top_n=None):

        nrows_str = str(nrows) if nrows is not None else "full"
        if augment_labels: nrows_str += "_augmented"
        if top_n is not None: nrows_str += "_" + str(top_n)

        path = os.path.join(DATA_DIR, "data_{}.pkl".format(nrows_str))

        if not os.path.isfile(path):
            notes = pd.read_csv(Tables.note_events, compression="gzip",
                                error_bad_lines=False, nrows=nrows)

            icd = pd.read_csv(Tables.diagnoses_icd, compression="gzip",
                              error_bad_lines=False)

            # Only get discharge notes
            discharge_notes = notes[notes[Keys.note_category] ==
                                    "Discharge summary"]

            dat = [[row[Keys.text],
                    (icd[icd[Keys.admission_id] ==
                         row[Keys.admission_id]])
                    [Keys.icd9].astype(str).tolist()]
                   for index, row in discharge_notes.iterrows()]

            # Do not train embeddings with X, use X_train because it will
            # cause data snooping.
            self.X, self.y_raw = zip(*dat)
            if top_n is not None: self._top_labels(top_n)
            if augment_labels: self._augment_labels()
            self._create_train_test_dataset()

            self.binarizer = MultiLabelBinarizer()
            self.y_train = self.binarizer.fit_transform(self.y_train_raw)
            self.y_test = self.binarizer.transform(self.y_test_raw)
            self.labels = self.binarizer.classes_

            self.y_train_graph = self._create_network_graph()

            # Pickle saving
            file = open(path, "wb")
            pickle.dump(self.X, file)
            pickle.dump(self.y_raw, file)
            pickle.dump(self.binarizer, file)
            pickle.dump(self.y_train_graph, file)

        else:
            # Pickle loading
            file = open(path, "rb")
            self.X = pickle.load(file)
            self.y_raw = pickle.load(file)
            self.binarizer = pickle.load(file)
            self.y_train_graph = pickle.load(file)

            self._create_train_test_dataset()
            self.y_train = self.binarizer.transform(self.y_train_raw)
            self.y_test = self.binarizer.transform(self.y_test_raw)
            self.labels = self.binarizer.classes_

    def _create_network_graph(self):
        """
        Creates label co-occurence graph as a networkx Graph based on the
        training dataset's labels.

        :return: G - networkx graph of label co-occurences
        """
        logging.info("Going to create label network graph with {}"
                     "samples.".format(len(self.y_train)))
        G = nx.Graph()
        for i, row in enumerate(self.y_train):

            if i % 1000 == 0:
                logging.info("Processing row: {}".format(i))

            pos = np.argwhere(row > 0).flatten()
            edge_tuples = list(product(pos, pos))
            for tuple in edge_tuples:
                if G.has_edge(tuple[0], tuple[1]):
                    G[tuple[0]][tuple[1]]['weight'] += 1
                else:
                    G.add_edge(tuple[0], tuple[1], weight=1)

        logging.info("Finished graph construction.")
        return G

    def _augment_labels(self):

        logging.info("Augmenting labels")
        codes_path = os.path.join(DATA_DIR, "codes.json")
        tree = ICD9(codes_path)

        old_average = np.average([len(row) for row in self.y_raw])

        new_y_raw = []

        for row in self.y_raw:
            new_row = set(row)

            for code in row:

                if len(code) > 3:
                    node = tree.find(code[:3] + "." + code[3:])

                    if node is None:
                        node = tree.find(code[:3] + "." + code[3])
                else:
                    node = tree.find(code)

                if node is not None:
                    for parent in node.parents:
                        new_row.add(parent.code.replace(".", ""))

            new_row.remove("ROOT")
            new_y_raw.append(list(new_row))

        self.y_raw = new_y_raw
        new_average = np.average([len(row) for row in self.y_raw])
        logging.info("Label augmentation complete.\nOld average label "
                     "count per document: {}\nNew average label count per "
                     "document {}".format(old_average, new_average))

    def _top_labels(self, n):
        """
        Filters out labels such that only top n labels in terms of frequency
        remains
        """
        counter = Counter(np.hstack(self.y_raw))
        most_common_labels = set(list(zip(*counter.most_common(n)))[0])
        self.y_raw = [[num for num in arr if num in most_common_labels]
                      for arr in self.y_raw]

    def _create_train_test_dataset(self):
        """
        Creates train and test split for MIMIC-III data. Ignores the labels
        which are not included in the training dataset.
        """
        self.X_train, self.X_test, self.y_train_raw, self.y_test_raw = \
            train_test_split(self.X,
                             self.y_raw,
                             test_size=0.1,
                             random_state=42)

        # Get all unique labels in the training
        train_icd = set(np.unique(np.hstack(self.y_train_raw)))
        logging.info("Training set have {} unique ICD9 codes."
                     .format(len(train_icd)))

        # Remove labels which are not included in the training dataset
        self.y_test_raw = [[num for num in arr if num in train_icd]
                           for arr in self.y_test_raw]


if __name__ == "__main__":
    data = Dataset(nrows=1000, top_n=100)
