import sys
import logging
import scipy
import numpy as np
import pandas as pd

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.sar import SAR
from recommenders.utils.notebook_utils import store_metadata

print(f"System version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")

# Top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = "100k"

# set log level to INFO
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def run_model():
    data = movielens.load_pandas_df(
        size=MOVIELENS_DATA_SIZE,
        header=["UserId", "MovieId", "Rating", "Timestamp"],
        title_col="Title",
    )

    # Convert the float precision to 32-bit in order to reduce memory consumption
    data["Rating"] = data["Rating"].astype(np.float32)

    data.head()

    # #%% md
    # ### 3.2 Split the data using the python random splitter provided in utilities:
    # # We split the full dataset into a `train` and `test` dataset to evaluate performance of the algorithm
    # against a held-out set not seen during training. Because SAR generates recommendations
    # based on user preferences, all users that are in the test set must also exist in the training set.
    # For this case, we can use the provided `python_stratified_split`
    # function which holds out a percentage (in this case 25%) of items from each user,
    # but ensures all users are in both `train` and `test` datasets.
    # Other options are available in the `dataset.python_splitters`
    # module which provide more control over how the split occurs.

    header = {
        "col_user": "UserId",
        "col_item": "MovieId",
        "col_rating": "Rating",
        "col_timestamp": "Timestamp",
        "col_prediction": "Prediction",
    }

    train, test = python_stratified_split(
        data, ratio=0.75, col_user=header["col_user"], col_item=header["col_item"], seed=42
    )

    model = SAR(
        similarity_type="jaccard",
        time_decay_coefficient=30,
        time_now=None,
        timedecay_formula=True,
        **header
    )

    model.fit(train)

    top_k = model.recommend_k_items(test, top_k=TOP_K, remove_seen=True)

    top_k_with_titles = top_k.join(
        data[["MovieId", "Title"]].drop_duplicates().set_index("MovieId"),
        on="MovieId",
        how="inner",
    ).sort_values(by=["UserId", "Prediction"], ascending=False)

    top_k_with_titles.head(10)

    ### 3.3 Evaluate the results

    # all ranking metrics have the same arguments
    args = [test, top_k]
    kwargs = dict(
        col_user="UserId",
        col_item="MovieId",
        col_rating="Rating",
        col_prediction="Prediction",
        relevancy_method="top_k",
        k=TOP_K,
    )

    eval_map = map_at_k(*args, **kwargs)
    eval_ndcg = ndcg_at_k(*args, **kwargs)
    eval_precision = precision_at_k(*args, **kwargs)
    eval_recall = recall_at_k(*args, **kwargs)

    print(f"Model:",
          f"Top K:\t\t {TOP_K}",
          f"MAP:\t\t {eval_map:f}",
          f"NDCG:\t\t {eval_ndcg:f}",
          f"Precision@K:\t {eval_precision:f}",
          f"Recall@K:\t {eval_recall:f}", sep='\n')

    # Record results for tests - ignore this cell
    store_metadata("map", eval_map)
    store_metadata("ndcg", eval_ndcg)
    store_metadata("precision", eval_precision)
    store_metadata("recall", eval_recall)


