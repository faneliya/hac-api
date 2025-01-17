import os
import sys
import torch
import cornac
import pandas as pd

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.evaluation.python_evaluation import (
    map,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.utils.notebook_utils import store_metadata

#from models.recommender_models.model_collaborative_filtering.m02_deep_dive_baseline import run_model
import util.constant as ENV


def run_model():

    print("m03_deep_dive_cornac_bivae")
    print(f"System version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Cornac version: {cornac.__version__}")

    # Select MovieLens data size: 100k, 1m, 10m, or 20m
    MOVIELENS_DATA_SIZE = '100k'
    # top k items to recommend
    TOP_K = 10

    # Model parameters
    LATENT_DIM = 50
    ENCODER_DIMS = [100]
    ACT_FUNC = "tanh"
    LIKELIHOOD = "pois"
    NUM_EPOCHS = 500
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001


    if os.path.isfile(f"{ENV.DATA_PATH}/m03/movielens.pickle"):
        data = pd.read_pickle(f"{ENV.DATA_PATH}/m03/movielens.pickle")
    else:
        data = movielens.load_pandas_df(
            size=MOVIELENS_DATA_SIZE,
            header=["userID", "itemID", "rating"]
        )
        data.to_pickle(f"{ENV.DATA_PATH}/m08/movielens.pickle")
        print(data.head())

    train, test = python_random_split(data, 0.75)
    train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)

    print('Number of users: {}'.format(train_set.num_users))
    print('Number of items: {}'.format(train_set.num_items))

    bivae = cornac.models.BiVAECF(
        k=LATENT_DIM,
        encoder_structure=ENCODER_DIMS,
        act_fn=ACT_FUNC,
        likelihood=LIKELIHOOD,
        n_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        use_gpu=torch.cuda.is_available(),
        verbose=True
    )

    with Timer() as t:
        bivae.fit(train_set)
    print("Took {} seconds for training.".format(t))

    with Timer() as t:
        all_predictions = predict_ranking(bivae, train, usercol='userID', itemcol='itemID', remove_seen=True)
    print("Took {} seconds for prediction.".format(t))

    eval_map = map(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

    print("MAP:\t%f" % eval_map,
          "NDCG:\t%f" % eval_ndcg,
          "Precision@K:\t%f" % eval_precision,
          "Recall@K:\t%f" % eval_recall, sep='\n')

    # Record results for tests - ignore this cell
    store_metadata("map", eval_map)
    store_metadata("ndcg", eval_ndcg)
    store_metadata("precision", eval_precision)
    store_metadata("recall", eval_recall)
