import os.path
import sys
import itertools
import pandas as pd

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.datasets.pandas_df_utils import filter_by
from recommenders.evaluation.python_evaluation import (
    rmse,
    mae,
    rsquared,
    exp_var,
    map,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.utils.notebook_utils import store_metadata

print(f"System version: {sys.version}")
print(f"Pandas version: {pd.__version__}")


def run_model():

    MOVIELENS_DATA_SIZE = "100k"
    TOP_K = 10

    if(os.path.isfile("./data/m02/movielens.pickle")):
        data = pd.read_pickle("./data/m02/movielens.pickle")
    else:
        data = movielens.load_pandas_df(
            size=MOVIELENS_DATA_SIZE,
            header=["UserId", "MovieId", "Rating", "Timestamp"]
        )
        data.to_pickle("./data/m02/movielens.pickle")


    print(data.head())

    train, test = python_random_split(data, ratio=0.75, seed=42)

    # Calculate avg ratings from the training set
    users_ratings = train.groupby(["UserId"])["Rating"].mean()
    users_ratings = users_ratings.to_frame().reset_index()
    users_ratings.rename(columns={"Rating": "AvgRating"}, inplace=True)

    users_ratings.head()

    # Generate prediction for the test set
    baseline_predictions = pd.merge(test, users_ratings, on=["UserId"], how="inner")

    baseline_predictions.loc[baseline_predictions["UserId"] == 1].head()

    baseline_predictions = baseline_predictions[["UserId", "MovieId", "AvgRating"]]

    cols = {
        "col_user": "UserId",
        "col_item": "MovieId",
        "col_rating": "Rating",
        "col_prediction": "AvgRating",
    }

    eval_rmse = rmse(test, baseline_predictions, **cols)
    eval_mae = mae(test, baseline_predictions, **cols)
    eval_rsquared = rsquared(test, baseline_predictions, **cols)
    eval_exp_var = exp_var(test, baseline_predictions, **cols)

    print("RMSE:\t\t%f" % eval_rmse,
          "MAE:\t\t%f" % eval_mae,
          "rsquared:\t%f" % eval_rsquared,
          "exp var:\t%f" % eval_exp_var, sep='\n')

    item_counts = train["MovieId"].value_counts().to_frame().reset_index()
    item_counts.columns = ["MovieId", "Count"]
    item_counts.head()

    user_item_col = ["UserId", "MovieId"]

    # Cross join users and items
    test_users = test['UserId'].unique()
    user_item_list = list(itertools.product(test_users, item_counts['MovieId']))
    users_items = pd.DataFrame(user_item_list, columns=user_item_col)

    print("Number of user-item pairs:", len(users_items))

    # Remove seen items (items in the train set) as we will not recommend those again to the users
    users_items_remove_seen = filter_by(users_items, train, user_item_col)

    print("After remove seen items:", len(users_items_remove_seen))

    # Generate recommendations
    baseline_recommendations = pd.merge(item_counts, users_items_remove_seen, on=['MovieId'], how='inner')
    baseline_recommendations.head()

    cols["col_prediction"] = "Count"

    eval_map = map(test, baseline_recommendations, k=TOP_K, **cols)
    eval_ndcg = ndcg_at_k(test, baseline_recommendations, k=TOP_K, **cols)
    eval_precision = precision_at_k(test, baseline_recommendations, k=TOP_K, **cols)
    eval_recall = recall_at_k(test, baseline_recommendations, k=TOP_K, **cols)

    print("MAP:\t%f" % eval_map,
          "NDCG@K:\t%f" % eval_ndcg,
          "Precision@K:\t%f" % eval_precision,
          "Recall@K:\t%f" % eval_recall, sep='\n')

    # Record results for tests - ignore this cell
    store_metadata("map", eval_map)
    store_metadata("ndcg", eval_ndcg)
    store_metadata("precision", eval_precision)
    store_metadata("recall", eval_recall)
    store_metadata("rmse", eval_rmse)
    store_metadata("mae", eval_mae)
    store_metadata("exp_var", eval_exp_var)
    store_metadata("rsquared", eval_rsquared)


