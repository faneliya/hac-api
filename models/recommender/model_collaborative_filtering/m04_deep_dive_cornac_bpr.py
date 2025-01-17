import os
import sys
import cornac
import pandas as pd

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.utils.notebook_utils import store_metadata
import util.constant as ENV

def run_model():

    print("m04_deep_dive_cornac_bpr")
    print(f"System version: {sys.version}")
    print(f"Cornac version: {cornac.__version__}")

    # Select MovieLens data size: 100k, 1m, 10m, or 20m
    MOVIELENS_DATA_SIZE = '100k'

    # top k items to recommend
    TOP_K = 10

    # Model parameters
    NUM_FACTORS = 200
    NUM_EPOCHS = 100

    if os.path.isfile(f"{ENV.DATA_PATH}/m04/movielens.pickle"):
        data = pd.read_pickle(f"{ENV.DATA_PATH}/m04/movielens.pickle")
    else:
        data = movielens.load_pandas_df(
            size=MOVIELENS_DATA_SIZE,
            header=["userID", "itemID", "rating"]
        )
        data.to_pickle(f"{ENV.DATA_PATH}/m04/movielens.pickle")

    data.head()
    train, test = python_random_split(data, 0.75)

    if os.path.isfile(f"{ENV.DATA_PATH}/m04/train_set.pickle"):
        train_set = pd.read_pickle(f"{ENV.DATA_PATH}/m04/train_set.pickle")
    else:
        train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)
        print(type(train_set))
        print(train_set)
        # train_set.to_pickle(f"{ENV.DATA_PATH}/m04/train_set.pickle")

    print('Number of users: {}'.format(train_set.num_users))
    print('Number of items: {}'.format(train_set.num_items))

    bpr = cornac.models.BPR(
        k=NUM_FACTORS,
        max_iter=NUM_EPOCHS,
        learning_rate=0.01,
        lambda_reg=0.001,
        verbose=True,
        seed=SEED
    )

    with Timer() as t:
        bpr.fit(train_set)
    print("Took {} seconds for training.".format(t))

    bpr.save(f'{ENV.DATA_PATH}/m04/bpr_model.pkl')
    bpr.load(f'{ENV.DATA_PATH}/m04/bpr_model.pkl/BPR/2025-01-04_12-01-43-361423.pkl')

    with Timer() as t:
        all_predictions = predict_ranking(bpr, train, usercol='userID', itemcol='itemID', remove_seen=True)
    print("Took {} seconds for prediction.".format(t))

    all_predictions.head()

    k = 10
    eval_map = map(test, all_predictions, col_prediction='prediction', k=k)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=k)
    eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=k)
    eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=k)

    print("MAP:\t%f" % eval_map,
          "NDCG:\t%f" % eval_ndcg,
          "Precision@K:\t%f" % eval_precision,
          "Recall@K:\t%f" % eval_recall, sep='\n')

    # Record results for tests - ignore this cell
    store_metadata("map", eval_map)
    store_metadata("ndcg", eval_ndcg)
    store_metadata("precision", eval_precision)
    store_metadata("recall", eval_recall)