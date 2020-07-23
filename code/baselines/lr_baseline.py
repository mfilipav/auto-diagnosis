import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../../")

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import constants
from utils.data_manager import Dataset
from utils.utils import report_evaluation

dataset = Dataset(nrows=constants.Dataset.nrows,
                  augment_labels=constants.Dataset.augment_labels,
                  top_n=constants.Dataset.top_n)

vectorizer = TfidfVectorizer(sublinear_tf=True,
                             max_features=constants.NLP.vocab_size,
                             stop_words='english')


X_train_v = vectorizer.fit_transform(dataset.X_train)
X_test_v = vectorizer.transform(dataset.X_test)

clf = OneVsRestClassifier(LogisticRegression(), n_jobs=8)

clf.fit(X_train_v, dataset.y_train)
yhat_test_raw = clf.predict_proba(X_test_v)

save = False

if save:
    with open('y_pred.pkl', 'wb') as f:
        pickle.dump(np.asarray(yhat_test_raw), f, pickle.HIGHEST_PROTOCOL)

yhat_test = np.array(yhat_test_raw >=
                     constants.NeuralNetworkTraining
                     .threshold).astype(np.int64)

report_evaluation(yhat_test, dataset.y_test, yhat_raw=yhat_test_raw)
