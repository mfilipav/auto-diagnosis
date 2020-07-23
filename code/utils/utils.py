import datetime
import inspect
import logging
import os
import sys
from collections import OrderedDict

sys.path.insert(0, "../")
sys.path.insert(0, "../code")

from texttable import Texttable
import constants
from utils import evaluation


def report_evaluation(y_hat, y, yhat_raw=None):
    """
    Calculates evaluation metrics, logs them and saves them to a results file.

    :param y_hat: binary predictions matrix
    :param y: binary ground truth matrix
    :param yhat_raw: prediction scores matrix (floats)
    :return: None
    """
    eval_metrics = evaluation.all_metrics(y_hat, y, yhat_raw=yhat_raw)
    eval_keys = sorted(eval_metrics.keys())

    ordered_eval_metrics = OrderedDict()
    for metric in eval_keys:
        ordered_eval_metrics[metric] = eval_metrics[metric]

    logging.info(str(ordered_eval_metrics))

    config_dict = OrderedDict()
    classes = inspect.getmembers(constants, inspect.isclass)
    for class_ in classes:
        for vars_, val in vars(class_[1]).items():
            if "__" not in vars_ and \
                    class_[0] != "Keys" and \
                    class_[0] != "Tables":
                config_dict[class_[0] + "_" + vars_] = val

    t = Texttable()
    t.set_precision(5)
    t.add_rows([["Configuration", "Value"]] +
               list(config_dict.items()))

    t2 = Texttable()
    t2.set_precision(5)
    t2.add_rows([["Evaluation Metric", "Value"]] +
                list(ordered_eval_metrics.items()))

    out = t.draw() + "\n" + t2.draw()

    ts_now = 'results-{:%Y-%m-%d--%H:%M:%S}.txt'.format(datetime.datetime.now())
    res_path = os.path.join(constants.LOGS_DIR, ts_now)

    out_file = open(res_path, "w+")
    out_file.write(out)
    out_file.close()
