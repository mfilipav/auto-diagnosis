import logging
import os

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
EMBEDDINGS_DIR = os.path.join(PROJECT_DIR, "embeddings")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

class Tables:
    note_events = os.path.join(DATA_DIR, "NOTEEVENTS.csv.gz")
    diagnoses_icd = os.path.join(DATA_DIR, "DIAGNOSES_ICD.csv.gz")
    procedures = os.path.join(DATA_DIR, "PROCEDURES_ICD.csv.gz")
    lab_events = os.path.join(DATA_DIR, "LABEVENTS.csv.gz")
    admissions = os.path.join(DATA_DIR, "ADMISSIONS.csv.gz")


class Keys:
    admission_id = "HADM_ID"
    icd9 = "ICD9_CODE"
    note_category = "CATEGORY"
    text = "TEXT"


class NLP:
    vocab_size = 40000
    sentence_length = 2500
    embedding_size = 300


class Deepwalk:
    dimensions = 256
    embedding_path = os.path.join(EMBEDDINGS_DIR, "icd_embedding.csv")
    attention_path = os.path.join(EMBEDDINGS_DIR, "icd_attention.csv")
    epochs = 9000
    window_size = 5
    beta = 0.1
    learning_rate = 0.005


class NeuralNetworkTraining:
    learning_rate = 0.0001
    batch_size = 32
    epochs = 30
    threshold = 0.35


class MLKNN:
    k = 3
    s = 0.5

class Dataset:
    nrows = None
    augment_labels = False
    top_n = None
