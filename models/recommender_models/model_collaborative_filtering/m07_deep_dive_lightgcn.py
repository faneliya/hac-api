import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.utils.notebook_utils import store_metadata

print(f"System version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"Tensorflow version: {tf.__version__}")

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Model parameters
EPOCHS = 50
BATCH_SIZE = 1024

SEED = DEFAULT_SEED  # Set None for non-deterministic results

yaml_file = "../../recommenders/models/deeprec/config/lightgcn.yaml"
user_file = "../../tests/resources/deeprec/lightgcn/user_embeddings.csv"
item_file = "../../tests/resources/deeprec/lightgcn/item_embeddings.csv"


def run_model():
    df = movielens.load_pandas_df(size=MOVIELENS_DATA_SIZE)

    print(df.head())

    train, test = python_stratified_split(df, ratio=0.75)

    data = ImplicitCF(train=train, test=test, seed=SEED)

    hparams = prepare_hparams(yaml_file,
                              n_layers=3,
                              batch_size=BATCH_SIZE,
                              epochs=EPOCHS,
                              learning_rate=0.005,
                              eval_epoch=5,
                              top_k=TOP_K,
                             )

    model = LightGCN(hparams, data, seed=SEED)

    with Timer() as train_time:
        model.fit()

    print("Took {} seconds for training.".format(train_time.interval))


    topk_scores = model.recommend_k_items(test, top_k=TOP_K, remove_seen=True)

    topk_scores.head()

    eval_map = map(test, topk_scores, k=TOP_K)
    eval_ndcg = ndcg_at_k(test, topk_scores, k=TOP_K)
    eval_precision = precision_at_k(test, topk_scores, k=TOP_K)
    eval_recall = recall_at_k(test, topk_scores, k=TOP_K)

    print("MAP:\t%f" % eval_map,
          "NDCG:\t%f" % eval_ndcg,
          "Precision@K:\t%f" % eval_precision,
          "Recall@K:\t%f" % eval_recall, sep='\n')

    # Record results for tests - ignore this cell
    store_metadata("map", eval_map)
    store_metadata("ndcg", eval_ndcg)
    store_metadata("precision", eval_precision)
    store_metadata("recall", eval_recall)

    model.infer_embedding(user_file, item_file)
