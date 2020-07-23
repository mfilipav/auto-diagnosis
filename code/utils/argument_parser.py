import argparse
import os
import sys

import constants

sys.path.insert(0, "../")
sys.path.insert(0, "../code")


def get_arg_parser(embedding_classifier=True):
    """
    Generates an argument parser for classification.py and
    embedding_classification.py

    :param embedding_classifier: boolean to set whether the multi_label
    classification process requires graph embeddings or not.

    :return: ArgumentParser object
    """
    parser = argparse.ArgumentParser()

    embedding_model_name = "word2vec_embeddings_{}.model" \
        .format(constants.NLP.embedding_size)

    parser.add_argument('--train_word2vec',
                        help='Use this flag if you want train word2vec'
                             'embeddings from scratch, or embeddings are '
                             'going to be loaded from {}'
                        .format(os.path.join(constants.EMBEDDINGS_DIR,
                                             embedding_model_name)),
                        action='store_true')

    if embedding_classifier:
        parser.add_argument('--train_attentionwalk',
                            help='Use this flag if you want train attentionwalk'
                                 'label embeddings from scratch, or embeddings '
                                 'are going to be loaded from {}'
                            .format(constants.Deepwalk.embedding_path),
                            action='store_true')

    parser.add_argument("-m",
                        "--model_name",
                        choices=("CAML", "CNN", "RNN"),
                        help="Model for the regressor",
                        action="store",
                        default="CAML"
                        )

    parser.add_argument("-g",
                        "--gpu",
                        help="gpu",
                        action="store",
                        default=None
                        )

    return parser
