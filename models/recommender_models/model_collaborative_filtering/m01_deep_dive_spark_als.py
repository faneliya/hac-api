import sys
import surprise

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import (
    rmse,
    mae,
    rsquared,
    exp_var,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    get_top_k_items,
)
from recommenders.models.surprise.surprise_utils import (
    predict,
    compute_ranking_predictions,
)
from recommenders.utils.notebook_utils import store_metadata


print(f"System version: {sys.version}")
print(f"Surprise version: {surprise.__version__}")

# Top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = "100k"


def run_model():
    data = movielens.load_pandas_df(
        size=MOVIELENS_DATA_SIZE, header=["userID", "itemID", "rating"]
    )

    data.head()

    train, test = python_random_split(data, 0.75)

    # 'reader' is being used to get rating scale (for MovieLens, the scale is [1, 5]).
    # 'rating_scale' parameter can be used instead for the later version of surprise lib:
    # https://github.com/NicolasHug/Surprise/blob/master/surprise/dataset.py
    train_set = surprise.Dataset.load_from_df(
        train, reader=surprise.Reader("ml-100k")
    ).build_full_trainset()

    train_set

    svd = surprise.SVD(random_state=0, n_factors=200, n_epochs=30, verbose=True)

    with Timer() as train_time:
        svd.fit(train_set)

    print(f"Took {train_time.interval} seconds for training.")


    predictions = predict(svd, test, usercol="userID", itemcol="itemID")
    predictions.head()

    with Timer() as test_time:
        all_predictions = compute_ranking_predictions(
            svd, train, usercol="userID", itemcol="itemID", remove_seen=True
        )

    print(f"Took {test_time.interval} seconds for prediction.")

    all_predictions.head()

    eval_rmse = rmse(test, predictions)
    eval_mae = mae(test, predictions)
    eval_rsquared = rsquared(test, predictions)
    eval_exp_var = exp_var(test, predictions)

    eval_map = map_at_k(test, all_predictions, col_prediction="prediction", k=TOP_K)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction="prediction", k=TOP_K)
    eval_precision = precision_at_k(
        test, all_predictions, col_prediction="prediction", k=TOP_K
    )
    eval_recall = recall_at_k(test, all_predictions, col_prediction="prediction", k=TOP_K)


    print(
        "RMSE:\t\t%f" % eval_rmse,
        "MAE:\t\t%f" % eval_mae,
        "rsquared:\t%f" % eval_rsquared,
        "exp var:\t%f" % eval_exp_var,
        sep="\n",
    )

    print("----")

    print(
        "MAP:\t\t%f" % eval_map,
        "NDCG:\t\t%f" % eval_ndcg,
        "Precision@K:\t%f" % eval_precision,
        "Recall@K:\t%f" % eval_recall,
        sep="\n",
    )

    # Record results for tests - ignore this cell
    store_metadata("rmse", eval_rmse)
    store_metadata("mae", eval_mae)
    store_metadata("rsquared", eval_rsquared)
    store_metadata("exp_var", eval_exp_var)
    store_metadata("map", eval_map)
    store_metadata("ndcg", eval_ndcg)
    store_metadata("precision", eval_precision)
    store_metadata("recall", eval_recall)
    store_metadata("train_time", train_time.interval)
    store_metadata("test_time", test_time.interval)
