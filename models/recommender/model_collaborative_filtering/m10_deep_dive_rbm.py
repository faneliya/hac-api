import sys
import pandas as pd
import matplotlib.pyplot as plt

import logging
import numpy as np
import tensorflow as tf

# tf.get_logger().setLevel(logging.ERROR)

from recommenders.models.rbm.rbm import RBM
from recommenders.datasets.python_splitters import numpy_stratified_split
from recommenders.datasets.sparse import AffinityMatrix
from recommenders.utils.timer import Timer
from recommenders.utils.plot import line_graph
from recommenders.datasets import movielens
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

#For interactive mode only

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))


def ranking_metrics(
        data_size,
        data_true,
        data_pred,
        K
):
    eval_map = map_at_k(data_true, data_pred, col_user="userID", col_item="movieID",
                        col_rating="rating", col_prediction="prediction",
                        relevancy_method="top_k", k=K)

    eval_ndcg = ndcg_at_k(data_true, data_pred, col_user="userID", col_item="movieID",
                          col_rating="rating", col_prediction="prediction",
                          relevancy_method="top_k", k=K)

    eval_precision = precision_at_k(data_true, data_pred, col_user="userID", col_item="movieID",
                                    col_rating="rating", col_prediction="prediction",
                                    relevancy_method="top_k", k=K)

    eval_recall = recall_at_k(data_true, data_pred, col_user="userID", col_item="movieID",
                              col_rating="rating", col_prediction="prediction",
                              relevancy_method="top_k", k=K)

    df_result = pd.DataFrame(
        {"Dataset": data_size,
         "K": K,
         "MAP": eval_map,
         "nDCG@k": eval_ndcg,
         "Precision@k": eval_precision,
         "Recall@k": eval_recall,
         },
        index=[0]
    )

    return df_result

def run_model():

    MOVIELENS_DATA_SIZE = '100k'

    mldf_100k = movielens.load_pandas_df(
        size=MOVIELENS_DATA_SIZE,
        header=['userID', 'movieID', 'rating', 'timestamp']
    )

    mldf_100k.head()

    ###################################################
    MOVIELENS_DATA_SIZE = '1m'

    mldf_1m = movielens.load_pandas_df(
        size=MOVIELENS_DATA_SIZE,
        header=['userID', 'movieID', 'rating', 'timestamp']
    )

    mldf_1m.head()

    # to use standard names across the analysis
    header = {
        "col_user": "userID",
        "col_item": "movieID",
        "col_rating": "rating",
    }

    # instantiate the splitter
    am1m = AffinityMatrix(df=mldf_1m, **header)

    # obtain the sparse matrix
    X1m, _, _ = am1m.gen_affinity_matrix()

    #Next, we split the matrix above into train and test set sparse matrices
    #
    Xtr_1m, Xtst_1m = numpy_stratified_split(X1m)

    _, (ax1m, ax2m) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    ax1m.hist(Xtr_1m[Xtr_1m != 0], 5, density=True)
    ax1m.set_title('Train')
    ax1m.set(xlabel="ratings", ylabel="density")
    ax2m.hist(Xtst_1m[Xtst_1m != 0], 5, density=True)
    ax2m.set_title('Test')
    ax2m.set(xlabel="ratings", ylabel="density")

    am100k = AffinityMatrix(df=mldf_100k, **header)
    X100k, _, _ = am100k.gen_affinity_matrix()
    Xtr_100k, Xtst_100k = numpy_stratified_split(X100k)

    _, (ax1k, ax2k) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    ax1k.hist(Xtr_100k[Xtr_100k != 0], 5, density=True)
    ax1k.set_title('Train')
    ax1k.set(xlabel="ratings", ylabel="density")
    ax2k.hist(Xtst_100k[Xtst_100k != 0], 5, density=True)
    ax2k.set_title('Test')
    ax2k.set(xlabel="ratings", ylabel="density")

    model_1m = RBM(
        possible_ratings=np.setdiff1d(np.unique(Xtr_1m), np.array([0])),
        visible_units=Xtr_1m.shape[1],
        hidden_units=1200,
        training_epoch=30,
        minibatch_size=350,
        with_metrics=True
    )

    # Model Fit
    with Timer() as train_time:
        model_1m.fit(Xtr_1m)

    print("Took {:.2f} seconds for training.".format(train_time.interval))

    # Plot the train RMSE as a function of the epochs
    line_graph(values=model_1m.rmse_train, labels='train', x_name='epoch', y_name='rmse_train')

    # number of top score elements to be recommended
    K = 10

    # Model prediction on the test set Xtst.
    with Timer() as prediction_time:
        top_k_1m = model_1m.recommend_k_items(Xtst_1m)

    print("Took {:.2f} seconds for prediction.".format(prediction_time.interval))

    top_k_df_1m = am1m.map_back_sparse(top_k_1m, kind='prediction')
    test_df_1m = am1m.map_back_sparse(Xtst_1m, kind='ratings')

    rating_1m = ranking_metrics(
        data_size="mv 1m",
        data_true=test_df_1m,
        data_pred=top_k_df_1m,
        K=10)

    print(rating_1m)

    # 100k
    model_100k = RBM(
        possible_ratings=np.setdiff1d(np.unique(Xtr_100k), np.array([0])),
        visible_units=Xtr_100k.shape[1],
        hidden_units=600,
        training_epoch=30,
        minibatch_size=60,
        keep_prob=0.9,
        with_metrics=True
    )

    with Timer() as train_time:
        model_100k.fit(Xtr_100k)

    print("Took {:.2f} seconds for training.".format(train_time.interval))

    # Plot the train RMSE as a function of the epochs
    line_graph(values=model_100k.rmse_train, labels='train', x_name='epoch', y_name='rmse_train')

    # Model prediction on the test set Xtst.
    with Timer() as prediction_time:
        top_k_100k = model_100k.recommend_k_items(Xtst_100k)

    print("Took {:.2f} seconds for prediction.".format(prediction_time.interval))

    # to df
    top_k_df_100k = am100k.map_back_sparse(top_k_100k, kind='prediction')
    test_df_100k = am100k.map_back_sparse(Xtst_100k, kind='ratings')

    ### 4.2.1 Model evaluation
    #
    eval_100k = ranking_metrics(
        data_size="mv 100k",
        data_true=test_df_100k,
        data_pred=top_k_df_100k,
        K=10)

    print(eval_100k)
