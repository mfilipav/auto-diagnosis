import logging
import os
import sys

sys.path.insert(0, "../")

import gensim
import numpy as np

import constants
from utils.data_manager import Dataset
from utils.preprocessing import Preprocessing


def load_embedding(vocab, path, dim_embedding, vocab_size):
    logging.info("Loading external embeddings from %s" % path)

    model = gensim.models.Word2Vec.load(path)
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for tok, idx in vocab.items():
        if tok in model.wv.vocab:
            external_embedding[idx] = model.wv[tok]
            matches += 1
        else:
            logging.info("Token: '{}' could not be loaded".format(tok))
            external_embedding[idx] = np.random.uniform(low=-0.25,
                                                        high=0.25,
                                                        size=dim_embedding)

    logging.info("Token: 'PAD' could not be loaded")
    logging.info("%d words out of %d could be loaded" % (matches,
                                                         vocab_size))
    return external_embedding


def create_word_to_vec_embeddings(dataset,
                                  embedding_size=constants
                                  .NLP.embedding_size,
                                  window=5,
                                  min_count=3):
    """
    Creates word2vec embeddings using gensim library

    :param dataset: Tokenized array of documents
    :param embedding_size: Dimensionality of the word embedding
    :param window: Maximum distance between the current and predicted word
    within a sentence
    :param min_count: Ignores all words with total frequency lower than
    this

    :return: Word2vec model from gensim library
    """
    model = gensim.models.word2vec.Word2Vec(dataset,
                                            size=embedding_size,
                                            window=window,
                                            min_count=min_count,
                                            workers=16)

    model_name = "word2vec_embeddings_{}.model".format(embedding_size)
    model.save(os.path.join(constants.EMBEDDINGS_DIR, model_name))
    return model

def main():
    dataset = Dataset(nrows=constants.Dataset.nrows,
                      augment_labels=constants.Dataset.augment_labels,
                      top_n=constants.Dataset.top_n)
    analyzer = Preprocessing(dataset.X_train) \
        .featurizer.build_analyzer()

    docs = [analyzer(doc) for doc in dataset.X_train]
    create_word_to_vec_embeddings(docs)

if __name__ == "__main__":
    main()