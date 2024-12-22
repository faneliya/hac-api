import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.ml.tuning import CrossValidator
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import FloatType, IntegerType, LongType

from recommenders.datasets import movielens
from recommenders.utils.spark_utils import start_or_get_spark
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkRatingEvaluation
from recommenders.tuning.parameter_sweep import generate_param_grid
from recommenders.datasets.spark_splitters import spark_random_split

print(f"System version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"PySpark version: {pyspark.__version__}")

MOVIELENS_DATA_SIZE = "100k"

COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
COL_PREDICTION = "prediction"
COL_TIMESTAMP = "Timestamp"

schema = StructType(
    (
        StructField(COL_USER, IntegerType()),
        StructField(COL_ITEM, IntegerType()),
        StructField(COL_RATING, FloatType()),
        StructField(COL_TIMESTAMP, LongType()),
    )
)

RANK = 10
MAX_ITER = 15
REG_PARAM = 0.05
K = 10

spark = start_or_get_spark("ALS Deep Dive", memory="16g")
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")


def run_model():

    dfs = movielens.load_spark_df(spark=spark, size=MOVIELENS_DATA_SIZE, schema=schema)
    dfs_train, dfs_test = spark_random_split(dfs, ratio=0.75, seed=42)

    als = ALS(
        maxIter=MAX_ITER,
        rank=RANK,
        regParam=REG_PARAM,
        userCol=COL_USER,
        itemCol=COL_ITEM,
        ratingCol=COL_RATING,
        coldStartStrategy="drop"
    )

    model = als.fit(dfs_train)
    dfs_pred = model.transform(dfs_test).drop(COL_RATING)

    evaluations = SparkRatingEvaluation(
        dfs_test,
        dfs_pred,
        col_user=COL_USER,
        col_item=COL_ITEM,
        col_rating=COL_RATING,
        col_prediction=COL_PREDICTION
    )

    print(
        "RMSE score = {}".format(evaluations.rmse()),
        "MAE score = {}".format(evaluations.mae()),
        "R2 score = {}".format(evaluations.rsquared()),
        "Explained variance score = {}".format(evaluations.exp_var()),
        sep="\n"
    )

    # Get the cross join of all user-item pairs and score them.
    users = dfs_train.select(COL_USER).distinct()
    items = dfs_train.select(COL_ITEM).distinct()
    user_item = users.crossJoin(items)
    dfs_pred = model.transform(user_item)

    # Remove seen items.
    dfs_pred_exclude_train = dfs_pred.alias("pred").join(
        dfs_train.alias("train"),
        (dfs_pred[COL_USER] == dfs_train[COL_USER]) & (dfs_pred[COL_ITEM] == dfs_train[COL_ITEM]),
        how='outer'
    )

    dfs_pred_final = dfs_pred_exclude_train.filter(dfs_pred_exclude_train["train.Rating"].isNull()) \
        .select('pred.' + COL_USER, 'pred.' + COL_ITEM, 'pred.' + "prediction")

    dfs_pred_final.show()

    evaluations = SparkRankingEvaluation(
        dfs_test,
        dfs_pred_final,
        col_user=COL_USER,
        col_item=COL_ITEM,
        col_rating=COL_RATING,
        col_prediction=COL_PREDICTION,
        k=K
    )

    print(
        "Precision@k = {}".format(evaluations.precision_at_k()),
        "Recall@k = {}".format(evaluations.recall_at_k()),
        "NDCG@k = {}".format(evaluations.ndcg_at_k()),
        "Mean average precision = {}".format(evaluations.map_at_k()),
        sep="\n"
    )

    param_dict = {
        "rank": [10, 15, 20],
        "regParam": [0.001, 0.1, 1.0]
    }

    param_grid = generate_param_grid(param_dict)

    rmse_score = []

    for g in param_grid:
        als = ALS(
            userCol=COL_USER,
            itemCol=COL_ITEM,
            ratingCol=COL_RATING,
            coldStartStrategy="drop",
            **g
        )
        model = als.fit(dfs_train)
        dfs_pred = model.transform(dfs_test).drop(COL_RATING)
        evaluations = SparkRatingEvaluation(
            dfs_test,
            dfs_pred,
            col_user=COL_USER,
            col_item=COL_ITEM,
            col_rating=COL_RATING,
            col_prediction=COL_PREDICTION
        )
        rmse_score.append(evaluations.rmse())

    rmse_score = [float('%.4f' % x) for x in rmse_score]
    rmse_score_array = np.reshape(rmse_score, (len(param_dict["rank"]), len(param_dict["regParam"])))

    rmse_df = pd.DataFrame(data=rmse_score_array, index=pd.Index(param_dict["rank"], name="rank"),
                           columns=pd.Index(param_dict["regParam"], name="reg. parameter"))

    fig, ax = plt.subplots()
    sns.heatmap(rmse_df, cbar=False, annot=True, fmt=".4g")

    dfs_rec = model.recommendForAllUsers(10)

    dfs_rec.show(10)

    users = dfs_train.select(als.getUserCol()).distinct().limit(3)

    dfs_rec_subset = model.recommendForUserSubset(users, 10)
    dfs_rec_subset.show(10)
    # cleanup spark instance

    spark.stop()


run_model()
