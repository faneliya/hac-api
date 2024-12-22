import os
import sys
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.evaluation.python_evaluation import (
    map, ndcg_at_k, precision_at_k, recall_at_k
)
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.utils.notebook_utils import store_metadata

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))


# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Model parameters
EPOCHS = 100
BATCH_SIZE = 256

SEED = DEFAULT_SEED  # Set None for non-deterministic results


def run_model():

    df = movielens.load_pandas_df(
        size=MOVIELENS_DATA_SIZE,
        header=["userID", "itemID", "rating", "timestamp"]
    )

    df.head()

    train, test = python_chrono_split(df, 0.75)

    test = test[test["userID"].isin(train["userID"].unique())]
    test = test[test["itemID"].isin(train["itemID"].unique())]

    leave_one_out_test = test.groupby("userID").last().reset_index()

    train_file = "./train.csv"
    test_file = "./test.csv"
    leave_one_out_test_file = "./leave_one_out_test.csv"
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    leave_one_out_test.to_csv(leave_one_out_test_file, index=False)

    data = NCFDataset(train_file=train_file, test_file=leave_one_out_test_file, seed=SEED,
                      overwrite_test_file_full=True)

    model = NCF(
        n_users=data.n_users,
        n_items=data.n_items,
        model_type="NeuMF",
        n_factors=4,
        layer_sizes=[16, 8, 4],
        n_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        verbose=10,
        seed=SEED
    )

    with Timer() as train_time:
        model.fit(data)

    print("Took {} seconds for training.".format(train_time.interval))

    predictions = [[row.userID, row.itemID, model.predict(row.userID, row.itemID)]
                   for (_, row) in test.iterrows()]

    predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
    predictions.head()

    with Timer() as test_time:
        users, items, preds = [], [], []
        item = list(train.itemID.unique())
        for user in train.userID.unique():
            user = [user] * len(item)
            users.extend(user)
            items.extend(item)
            preds.extend(list(model.predict(user, item, is_list=True)))

        all_predictions = pd.DataFrame(data={"userID": users, "itemID": items, "prediction": preds})

        merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
        all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

    print("Took {} seconds for prediction.".format(test_time.interval))

    eval_map = map(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

    print("MAP:\t%f" % eval_map,
          "NDCG:\t%f" % eval_ndcg,
          "Precision@K:\t%f" % eval_precision,
          "Recall@K:\t%f" % eval_recall, sep='\n')

    k = TOP_K

    ndcgs = []
    hit_ratio = []

    for b in data.test_loader():
        user_input, item_input, labels = b
        output = model.predict(user_input, item_input, is_list=True)

        output = np.squeeze(output)
        rank = sum(output >= output[0])
        if rank <= k:
            ndcgs.append(1 / np.log(rank + 1))
            hit_ratio.append(1)
        else:
            ndcgs.append(0)
            hit_ratio.append(0)

    eval_ndcg = np.mean(ndcgs)
    eval_hr = np.mean(hit_ratio)

    print("HR:\t%f" % eval_hr)
    print("NDCG:\t%f" % eval_ndcg)

    ### 3.5.1 Training GMF and MLP model
    # model.save`, we can set the `dir_name` to store the parameters of GMF and MLP
    #
    model = NCF(
        n_users=data.n_users,
        n_items=data.n_items,
        model_type="GMF",
        n_factors=4,
        layer_sizes=[16, 8, 4],
        n_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        verbose=10,
        seed=SEED
    )

    with Timer() as train_time:
        model.fit(data)

    print("Took {} seconds for training.".format(train_time.interval))

    model.save(dir_name=".pretrain/GMF")

    model = NCF(
        n_users=data.n_users,
        n_items=data.n_items,
        model_type="MLP",
        n_factors=4,
        layer_sizes=[16, 8, 4],
        n_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        verbose=10,
        seed=SEED
    )

    with Timer() as train_time:
        model.fit(data)

    print("Took {} seconds for training.".format(train_time.interval))

    model.save(dir_name=".pretrain/MLP")

    ############################################################################
    ### 3.5.2 Load pre-trained GMF and MLP model for NeuMF
    # `model.load`, we can set the `gmf_dir` and `mlp_dir` to store the parameters for NeuMF.
    #
    model = NCF(
        n_users=data.n_users,
        n_items=data.n_items,
        model_type="NeuMF",
        n_factors=4,
        layer_sizes=[16, 8, 4],
        n_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        verbose=10,
        seed=SEED
    )

    model.load(gmf_dir=".pretrain/GMF", mlp_dir=".pretrain/MLP", alpha=0.5)

    with Timer() as train_time:
        model.fit(data)

    print("Took {} seconds for training.".format(train_time.interval))

    ### 3.5.3 Compare with not pre-trained NeuMF
    # You can use beforementioned evaluation methods to evaluate the pre-trained
    # `NCF` Model. Usually, we will find the performance of pre-trained NCF
    # is better than the not pre-trained.

    with Timer() as test_time:

        users, items, preds = [], [], []
        item = list(train.itemID.unique())
        for user in train.userID.unique():
            user = [user] * len(item)
            users.extend(user)
            items.extend(item)
            preds.extend(list(model.predict(user, item, is_list=True)))

        all_predictions = pd.DataFrame(data={"userID": users, "itemID": items, "prediction": preds})

        merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
        all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

    print("Took {} seconds for prediction.".format(test_time.interval))

    eval_map2 = map(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_ndcg2 = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_precision2 = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_recall2 = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

    print("MAP:\t%f" % eval_map2,
          "NDCG:\t%f" % eval_ndcg2,
          "Precision@K:\t%f" % eval_precision2,
          "Recall@K:\t%f" % eval_recall2, sep='\n')

    # Record results for tests - ignore this cell
    store_metadata("map", eval_map)
    store_metadata("ndcg", eval_ndcg)
    store_metadata("precision", eval_precision)
    store_metadata("recall", eval_recall)
    store_metadata("map2", eval_map2)
    store_metadata("ndcg2", eval_ndcg2)
    store_metadata("precision2", eval_precision2)
    store_metadata("recall2", eval_recall2)

    ### 3.5.4 Delete pre-trained directory
    save_dir = ".pretrain"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    print("Did \'%s\' exist?: %s" % (save_dir, os.path.exists(save_dir)))


