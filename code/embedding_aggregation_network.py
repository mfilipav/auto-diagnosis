import logging
import os
import sys

import pandas as pd

sys.path.insert(0, "../")
sys.path.insert(0, "../code")

import types
from skorch import NeuralNet
import torch
from torch import nn

import skorch
import numpy as np
import skorch.dataset

import constants
from models.CAML import CAML
from models.CNN import CNN
from models.RNN import RNN
import utils.embeddings
from attentionwalk.attentionwalk_util import tab_printer
from utils.data_manager import Dataset
from utils.preprocessing import Preprocessing
from utils.utils import report_evaluation
from utils.argument_parser import get_arg_parser

try:
    import setGPU

    logging.info(str(os.getenv("CUDA_VISIBLE_DEVICES")))
except ImportError:
    logging.warning("Could not import setGPU module")

F = nn.functional


def get_embedding_tensor(preprocessor):
    # Load embeddings (pre-trained from a file)
    # Get full path by joining name of file with directory taken from constants.py
    model_name = "word2vec_embeddings_{}.model" \
        .format(constants.NLP.embedding_size)
    full_path = os.path.join(constants.EMBEDDINGS_DIR, model_name)

    logging.info("Will load word embeddings from: {}".format(full_path))
    # Load embeddings
    vocab = preprocessor.featurizer.vocabulary_
    pretrained = utils.embeddings.load_embedding(vocab,
                                                 full_path,
                                                 constants.NLP.embedding_size,
                                                 constants.NLP.vocab_size)
    weights = torch.FloatTensor(pretrained)

    logging.info("Word embeddings are loaded.")
    return weights


def _generate_deepwalk_parameters(y_train_graph):
    """
    Generates an object which includes settings for the deepwalk
    algorithm and adds y_train_graph to it.

    :param y_train_graph: Networkx graph for label co-occurence
    graph.
    :return: args: an object which includes the settings for
    deepwalk algorithm
    """
    args = types.SimpleNamespace()
    args.embedding_path = constants.Deepwalk.embedding_path
    args.attention_path = constants.Deepwalk.attention_path
    args.dimensions = constants.Deepwalk.dimensions
    args.epochs = constants.Deepwalk.epochs
    args.window_size = constants.Deepwalk.window_size
    args.beta = constants.Deepwalk.beta
    args.learning_rate = constants.Deepwalk.learning_rate
    args.graph = y_train_graph
    args.device = 'cuda:' + str(os.getenv("CUDA_VISIBLE_DEVICES")) \
        if torch.cuda.is_available() else 'cpu'
    return args


def get_network_class(network_name):
    models = {
        "CAML": CAML,
        "CNN": CNN,
        "RNN": RNN
    }
    return models[network_name]


def run():
    parser = get_arg_parser(embedding_classifier=False)
    cmd_args = parser.parse_args()

    if cmd_args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cmd_args.gpu)
        gpunum = os.getenv('CUDA_VISIBLE_DEVICES')
        logging.info("GPU has been set to {}".format(gpunum))

    logging.info("Model used for the classification network: {}"
                 .format(cmd_args.model_name))

    # 1. Dataset retrieval
    # --------------------

    tab_printer(constants.Dataset)
    dataset = Dataset(nrows=constants.Dataset.nrows,
                      augment_labels=constants.Dataset.augment_labels,
                      top_n=constants.Dataset.top_n)

    logging.info("Going to create vocabulary and fit a preprocessing pipeline"
                 "using {} samples. Settings will be listed below"
                 .format(len(dataset.X_train)))

    # 2. Preprocessing
    # -----------------

    tab_printer(constants.NLP)
    preprocessor = Preprocessing(dataset.X_train)

    # Preprocess documents
    X_train = preprocessor.transform_documents(dataset.X_train)
    X_test = preprocessor.transform_documents(dataset.X_test)

    # 3. Word embeddings with word2vec
    # --------------------------------

    # Train word2vec embeddings if train_word2vec option
    # is selected
    if cmd_args.train_word2vec: utils.embeddings.main()
    weights = get_embedding_tensor(preprocessor)

    logging.info("Word embeddings are loaded.")

    # 4. Label Network Optim
    # -----------------------

    device = 'cuda:' + str(os.getenv("CUDA_VISIBLE_DEVICES")) \
        if torch.cuda.is_available() else 'cpu'
    logging.info("Going to run on device: {}".format(device))

    args = _generate_deepwalk_parameters(dataset.y_train_graph)
    label_embeddings = np.array(pd.read_csv(args.embedding_path).iloc[:, 1:].values)
    label_embeddings_weights = torch.FloatTensor(label_embeddings)

    label_network = NeuralNet(
        CAML,
        max_epochs=50,
        lr=constants.NeuralNetworkTraining.learning_rate,
        batch_size=constants.NeuralNetworkTraining.batch_size,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.BCEWithLogitsLoss,

        module__output_dim=dataset.y_train.shape[1],
        module__embedding=label_embeddings_weights,
        module__embedding_dim=args.dimensions,
        module__kernel_size=1,

        device=device,
        train_split=skorch.dataset.CVSplit(stratified=False),
    )

    label_network.fit(dataset.y_train, dataset.y_train.astype(np.float32))

    # 5. Evaluation
    # -------------

    yhat_test_raw_logits = label_network.predict_proba(dataset.y_test)
    yhat_test_raw = torch.sigmoid(torch.Tensor(yhat_test_raw_logits)).numpy()
    yhat_test = np.array(yhat_test_raw >=
                         constants.NeuralNetworkTraining.threshold) \
        .astype(np.int64)

    report_evaluation(yhat_test, dataset.y_test, yhat_raw=yhat_test_raw)


if __name__ == "__main__":
    run()
