import sys

sys.path.insert(0, "../")

import constants
from dstoolbox.transformers import Padder2d
from dstoolbox.transformers import TextFeaturizer
from sklearn.pipeline import Pipeline
import re
from utils.data_manager import Dataset
import nltk
import logging

nltk.download('punkt')


class Preprocessing:
    """
    Creates a pipeline fitted to X for the preprocessing of the medical
    records. It wraps TextFeaturizer and Padder2d classes from dstoolbox
    to remove english stopwords and numeric words, tokenize, pad (maximum
    length determined by constants.Config.sentence_length), and limit
    vocabulary size by constants.Config.vocab_size, and replace any token
    which is not included in the vocabulary by "UNK" token.

    Parameters
    ----------
    X : list
        List of medical discharge records from MIMIC-III database

    Attributes
    ----------
    featurizer : (~dstoolbox.transformers.TextFeaturizer)
        TextFeaturizer instance to analyze documents and  to look at the
        vocabulary which is trained on X

    pipeline : (~sklearn.pipeline import Pipeline)
        Pipeline object to transform datasets

    Notes
    -----

    Documents can be transformed by using transform method of the pipeline:
    transformed = preprocessor.transform_documents(dataset.X_train)


    TODO: Implement spell correction

    """

    def __init__(self, X):
        """
        Initializer of the Preprocessing class

        :param X: List of medical discharge records from MIMIC-III database
        :type X: list
        """
        self.featurizer = TextFeaturizer(max_features=
                                         constants.NLP.vocab_size - 2,
                                         unknown_token="UNK",
                                         stop_words="english",
                                         preprocessor=
                                         Preprocessing._preprocess_doc)

        steps = [
            ('to_idx', self.featurizer),
            ('pad', Padder2d(max_len=constants.NLP.sentence_length,
                             pad_value=constants.NLP.vocab_size - 1,
                             dtype=int)),
        ]

        self.pipeline = Pipeline(steps)
        logging.info("Going to fit text featurizer using {} records."
                     .format(len(X)))
        self.pipeline.fit(X)

    def preview_document_transform(self, document):
        """
        A method to test

        :param document:
        :type document: str
        """
        analyzer = self.featurizer.build_analyzer()
        vocab = self.featurizer.get_feature_names()
        tokens = self.featurizer.transform([document])[0]

        logging.info("Raw Document:\n" + document)
        logging.info("Analyzer output: \n" + str(analyzer(document)))
        logging.info("Pipeline output: \n" + str([vocab[tok] if
                                                  tok < len(vocab) else "PAD"
                                                  for tok in tokens]))

    def transform_documents(self, documents):
        return self.pipeline.transform(documents)

    @staticmethod
    def _preprocess_doc(doc):
        doc = doc[doc.find("Service:"):]
        doc = re.sub(r"\[\*\*.[^*]*\*\*\]", "", doc)
        doc = re.sub(r'[\d]+', "num", doc)
        doc = doc.lower()
        return doc


if __name__ == "__main__":
    dataset = Dataset(nrows=100)
    preprocessor = Preprocessing(dataset.X_train)
    preprocessor.preview_document_transform(dataset.X_test[0])
    transformed = preprocessor.transform_documents(dataset.X_train)

    # Example usage
    print(transformed[0])
