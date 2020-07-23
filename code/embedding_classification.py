import logging
import os
import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../code")

import types
from skorch import NeuralNet
import torch
from torch import nn

import numpy as np
import pandas as pd
from skmultilearn.adapt import MLkNN

import constants
from utils import evaluation
from models.CAML import CAML
from models.CNN import CNN
from models.RNN import RNN
import utils.embeddings
from attentionwalk.attentionwalk_util import tab_printer
from attentionwalk.attentionwalk import train_attention_walk
from utils.data_manager import Dataset
from utils.preprocessing import Preprocessing
from utils.argument_parser import get_arg_parser
from utils.utils import report_evaluation

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


def _get_label_embeddings(y, embeddings):
    y_embedded = []
    for row in y:
        pos = np.argwhere(row > 0).flatten()
        y_embedded.append(np.average(embeddings[pos], axis=0))
    return np.stack(y_embedded)


def get_network_class(network_name):
    models = {
        "CAML": CAML,
        "CNN": CNN,
        "RNN": RNN
    }
    return models[network_name]


def run():
    parser = get_arg_parser()
    cmd_args = parser.parse_args()

    if cmd_args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cmd_args.gpu)
        gpunum = os.getenv('CUDA_VISIBLE_DEVICES')
        logging.info("GPU has been set to {}".format(gpunum))

    logging.info("Model used for the regression network: {}"
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

    # Train word2vec embeddings if train_word2vec option is selected
    if cmd_args.train_word2vec: utils.embeddings.main()
    weights = get_embedding_tensor(preprocessor)

    # 4. Node embeddings with AttentionWalk
    # -------------------------------------
    args = _generate_deepwalk_parameters(dataset.y_train_graph)
    if cmd_args.train_attentionwalk: train_attention_walk(args)

    graph_embeddings = pd.read_csv(args.embedding_path).iloc[:, 1:].values

    # Get document representations using node embeddings
    y_embedded = _get_label_embeddings(dataset.y_train, graph_embeddings)
    y_test_embedded = _get_label_embeddings(dataset.y_test, graph_embeddings)

    # 5. Regressor Training
    # ---------------------

    device = 'cuda:' + str(os.getenv("CUDA_VISIBLE_DEVICES")) \
        if torch.cuda.is_available() else 'cpu'

    regressor_nn = NeuralNet(
        get_network_class(cmd_args.model_name),
        max_epochs=constants.NeuralNetworkTraining.epochs,
        lr=constants.NeuralNetworkTraining.learning_rate,
        batch_size=constants.NeuralNetworkTraining.batch_size,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.MSELoss,

        module__output_dim=args.dimensions,
        module__embedding=weights,
        module__embedding_dim=constants.NLP.embedding_size,

        device=device,
        train_split=None,
    )

    # Train the regressor neural network
    regressor_nn.fit(X_train, y_embedded.astype(np.float32))

    # 6. Train Multi-label KNN algorithm
    # ----------------------------------

    tab_printer(constants.MLKNN)

    # Train multi-label KNN to turn label embeddings into label predictions
    classifier = MLkNN(k=constants.MLKNN.k, s=constants.MLKNN.s)
    classifier.fit(y_embedded, dataset.y_train)

    # 7. Evaluation
    # -------------

    # Label prediction with documents
    y_test_pred = regressor_nn.predict(X_test)
    preds = classifier.predict(y_test_pred)
    preds_raw = classifier.predict_proba(y_test_pred)

    # Label prediction with label embeddings
    preds_w_labels = classifier.predict(y_test_embedded)
    preds_w_labels_raw = classifier.predict_proba(y_test_embedded)

    # Log evaluation result with label embeddings
    eval_metrics_w_labels = evaluation \
        .all_metrics(preds_w_labels.toarray(),
                     dataset.y_test,
                     yhat_raw=preds_w_labels_raw.toarray())

    logging.info(str(eval_metrics_w_labels))

    # Log evaluation result with documents
    report_evaluation(preds.toarray(),
                      dataset.y_test,
                      yhat_raw=preds_raw.toarray())


if __name__ == "__main__":
    run()
