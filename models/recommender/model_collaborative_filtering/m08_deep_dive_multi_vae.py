import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras

from tempfile import TemporaryDirectory
from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.split_utils import min_rating_filter_pandas
from recommenders.datasets.python_splitters import numpy_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.datasets.sparse import AffinityMatrix
from recommenders.utils.python_utils import binarize
from recommenders.models.vae.multinomial_vae import Mult_VAE

import util.constant as ENV
import warnings


def run_model():
    # 경고메세지 끄기
    warnings.filterwarnings(action='ignore')
    # 다시 출력하게 하기
    # warnings.filterwarnings(action='default')

    print(f"System version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Tensorflow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")

    # top k items to recommend
    TOP_K = 10
    # Select MovieLens data size: 100k, 1m, 10m, or 20m
    MOVIELENS_DATA_SIZE = '1m'
    # Model parameters
    HELDOUT_USERS = 600  # CHANGE FOR DIFFERENT DATASIZE
    INTERMEDIATE_DIM = 200
    LATENT_DIM = 70
    EPOCHS = 400
    EPOCHS = 200
    BATCH_SIZE = 100
    # temporary Path to save the optimal model's weights
    # tmp_dir = TemporaryDirectory()
    tmp_dir = f"{ENV.DATA_PATH}/m08"
    WEIGHTS_PATH_WITHOUT_ANNEAL = f"{tmp_dir}/mvae_weights_without_anneal.hdf5"
    WEIGHTS_PATH_WITH_ANNEAL = f"{tmp_dir}/mvae_weights_with_anneal.hdf5"
    WEIGHTS_PATH_OPTIMAL_BETA = f"{tmp_dir}/mvae_weights_optimal_beta.hdf5"

    print(tmp_dir)
    print(WEIGHTS_PATH_WITHOUT_ANNEAL)
    print(WEIGHTS_PATH_WITH_ANNEAL)
    print(WEIGHTS_PATH_OPTIMAL_BETA)

    SEED = 98765

    if os.path.isfile(f"{ENV.DATA_PATH}/m08/movielens.pickle"):
        df = pd.read_pickle(f"{ENV.DATA_PATH}/m08/movielens.pickle")
    else:
        df = movielens.load_pandas_df(
            size=MOVIELENS_DATA_SIZE,
            header=["userID", "itemID", "rating", "timestamp"]
        )
        df.to_pickle(f"{ENV.DATA_PATH}/m08/movielens.pickle")
    # df.head()
    # print(df.shape)

    # Binarize the data (only keep ratings >= 4)
    df_preferred = df[df['rating'] > 3.5]
    print(df_preferred.shape)
    df_low_rating = df[df['rating'] <= 3.5]
    # df.head()
    print(df_preferred.head(10))

    # Keep users who clicked on at least 5 movies
    df = min_rating_filter_pandas(df_preferred, min_rating=5, filter_by="user")
    # Keep movies that were clicked on by at least on 1 user
    df = min_rating_filter_pandas(df, min_rating=1, filter_by="item")
    # Obtain both usercount and itemcount after filtering
    usercount = df[['userID']].groupby('userID', as_index=False).size()
    itemcount = df[['itemID']].groupby('itemID', as_index=False).size()
    # Compute sparsity after filtering
    sparsity = 1. * df.shape[0] / (usercount.shape[0] * itemcount.shape[0])

    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (df.shape[0], usercount.shape[0], itemcount.shape[0], sparsity * 100))

    unique_users = sorted(df.userID.unique())
    np.random.seed(SEED)
    unique_users = np.random.permutation(unique_users)

    # Create train/validation/test users
    n_users = len(unique_users)
    print("Number of unique users:", n_users)
    train_users = unique_users[:(n_users - HELDOUT_USERS * 2)]
    print("Number of training users:", len(train_users))
    val_users = unique_users[(n_users - HELDOUT_USERS * 2): (n_users - HELDOUT_USERS)]
    print("Number of validation users:", len(val_users))
    test_users = unique_users[(n_users - HELDOUT_USERS):]
    print("Number of test users:", len(test_users))
    # For training set keep only users that are in train_users list
    train_set = df.loc[df['userID'].isin(train_users)]
    print("Number of training observations: ", train_set.shape[0])
    # For validation set keep only users that are in val_users list
    val_set = df.loc[df['userID'].isin(val_users)]
    print("Number of validation observations: ", val_set.shape[0])
    # For test set keep only users that are in test_users list
    test_set = df.loc[df['userID'].isin(test_users)]
    print("Number of test observations: ", test_set.shape[0])
    # train_set/val_set/test_set contain user - movie interactions with rating 4 or 5
    # Obtain list of unique movies used in training set
    unique_train_items = pd.unique(train_set['itemID'])
    print("Number of unique movies that rated in training set", unique_train_items.size)
    # For validation set keep only movies that used in training set
    val_set = val_set.loc[val_set['itemID'].isin(unique_train_items)]
    print("Number of validation observations after filtering: ", val_set.shape[0])
    # For test set keep only movies that used in training set
    test_set = test_set.loc[test_set['itemID'].isin(unique_train_items)]
    print("Number of test observations after filtering: ", test_set.shape[0])

    # Instantiate the sparse matrix generation for train, validation and test sets
    # use list of unique items from training set for all sets
    am_train = AffinityMatrix(df=train_set, items_list=unique_train_items)
    am_val = AffinityMatrix(df=val_set, items_list=unique_train_items)
    am_test = AffinityMatrix(df=test_set, items_list=unique_train_items)

    # Obtain the sparse matrix for train, validation and test sets
    train_data, _, _ = am_train.gen_affinity_matrix()
    # print(train_data.shape)
    val_data, val_map_users, val_map_items = am_val.gen_affinity_matrix()
    # print(val_data.shape)
    test_data, test_map_users, test_map_items = am_test.gen_affinity_matrix()
    # print(test_data.shape)

    # Split validation and test data into training and testing parts
    val_data_tr, val_data_te = numpy_stratified_split(val_data, ratio=0.75, seed=SEED)
    test_data_tr, test_data_te = numpy_stratified_split(test_data, ratio=0.75, seed=SEED)

    # Binarize train, validation and test data
    train_data = binarize(a=train_data, threshold=3.5)
    val_data = binarize(a=val_data, threshold=3.5)
    test_data = binarize(a=test_data, threshold=3.5)

    # Binarize validation data: training part
    val_data_tr = binarize(a=val_data_tr, threshold=3.5)

    # Binarize validation data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
    val_data_te_ratings = val_data_te.copy()
    val_data_te = binarize(a=val_data_te, threshold=3.5)

    # Binarize test data: training part
    test_data_tr = binarize(a=test_data_tr, threshold=3.5)

    # Binarize test data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
    test_data_te_ratings = test_data_te.copy()
    test_data_te = binarize(a=test_data_te, threshold=3.5)

    # retrieve real ratings from initial dataset

    test_data_te_ratings = pd.DataFrame(test_data_te_ratings)
    val_data_te_ratings = pd.DataFrame(val_data_te_ratings)

    for index, i in df_low_rating.iterrows():
        user_old = i['userID']  # old value
        item_old = i['itemID']  # old value

        if (test_map_users.get(user_old) is not None) and (test_map_items.get(item_old) is not None):
            user_new = test_map_users.get(user_old)  # new value
            item_new = test_map_items.get(item_old)  # new value
            rating = i['rating']
            test_data_te_ratings.at[user_new, item_new] = rating

        if (val_map_users.get(user_old) is not None) and (val_map_items.get(item_old) is not None):
            user_new = val_map_users.get(user_old)  # new value
            item_new = val_map_items.get(item_old)  # new value
            rating = i['rating']
            val_data_te_ratings.at[user_new, item_new] = rating

    val_data_te_ratings = val_data_te_ratings.to_numpy()
    test_data_te_ratings = test_data_te_ratings.to_numpy()
    # test_data_te_ratings

    # Just checking
    print(np.sum(val_data))
    print(np.sum(val_data_tr))
    print(np.sum(val_data_te))

    # Just checking
    print(np.sum(test_data))
    print(np.sum(test_data_tr))
    print(np.sum(test_data_te))

    ####################################################################################################################
    model_without_anneal = Mult_VAE(n_users=train_data.shape[0],  # Number of unique users in the training set
                                    original_dim=train_data.shape[1],  # Number of unique items in the training set
                                    intermediate_dim=INTERMEDIATE_DIM,
                                    latent_dim=LATENT_DIM,
                                    n_epochs=EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    k=TOP_K,
                                    verbose=0,
                                    seed=SEED,
                                    save_path=WEIGHTS_PATH_WITHOUT_ANNEAL,
                                    drop_encoder=0.5,
                                    drop_decoder=0.5,
                                    annealing=False,
                                    beta=1.0
                                    )

    # traing ###########################################################################################################
    with Timer() as t:
        model_without_anneal.fit(x_train=train_data,
                                 x_valid=val_data,
                                 x_val_tr=val_data_tr,
                                 x_val_te=val_data_te_ratings,
                                 mapper=am_val
                                 )
    print("Took {} seconds for training.".format(t))

    model_without_anneal.display_metrics()
    ndcg_val_without_anneal = model_without_anneal.ndcg_per_epoch()

    # Use k = 10
    with Timer() as t:
        # Model prediction on the training part of test set
        top_k = model_without_anneal.recommend_k_items(x=test_data_tr,
                                                       k=10,
                                                       remove_seen=True
                                                       )
        # Convert sparse matrix back to df
        top_k_df = am_test.map_back_sparse(top_k, kind='prediction')
        test_df = am_test.map_back_sparse(test_data_te_ratings,
                                          kind='ratings')  # use test_data_te_, with the original ratings

    print("Took {} seconds for prediction.".format(t))

    # Use the ranking metrics for evaluation ###########################################################################
    eval_map_1 = map_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_ndcg_1 = ndcg_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_precision_1 = precision_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_recall_1 = recall_at_k(test_df, top_k_df, col_prediction='prediction', k=10)

    print("MAP@10:\t\t%f" % eval_map_1,
          "NDCG@10:\t%f" % eval_ndcg_1,
          "Precision@10:\t%f" % eval_precision_1,
          "Recall@10: \t%f" % eval_recall_1, sep='\n')
    ####################################################################################################################

    # Use k = TOP_K
    with Timer() as t:
        # Model prediction on the training part of test set
        top_k = model_without_anneal.recommend_k_items(x=test_data_tr,
                                                       k=TOP_K,
                                                       remove_seen=True
                                                       )
        # Convert sparse matrix back to df
        top_k_df = am_test.map_back_sparse(top_k, kind='prediction')
        test_df = am_test.map_back_sparse(test_data_te_ratings,
                                          kind='ratings')  # use test_data_te_, with the original ratings

    print("Took {} seconds for prediction.".format(t))

    # Use the ranking metrics for evaluation ###########################################################################
    eval_map_2 = map_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_ndcg_2 = ndcg_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_precision_2 = precision_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_recall_2 = recall_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    print("MAP@100:\t%f" % eval_map_2,
          "NDCG@100:\t%f" % eval_ndcg_2,
          "Precision@100:\t%f" % eval_precision_2,
          "Recall@100: \t%f" % eval_recall_2, sep='\n')
    ####################################################################################################################
    # model -2
    model_with_anneal = Mult_VAE(n_users=train_data.shape[0],  # Number of unique users in the training set
                                 original_dim=train_data.shape[1],  # Number of unique items in the training set
                                 intermediate_dim=INTERMEDIATE_DIM,
                                 latent_dim=LATENT_DIM,
                                 n_epochs=EPOCHS,
                                 batch_size=BATCH_SIZE,
                                 k=TOP_K,
                                 verbose=0,
                                 seed=SEED,
                                 save_path=WEIGHTS_PATH_WITH_ANNEAL,
                                 drop_encoder=0.5,
                                 drop_decoder=0.5,
                                 annealing=True,
                                 anneal_cap=1.0,
                                 )
    # traing
    with Timer() as t:
        model_with_anneal.fit(x_train=train_data,
                              x_valid=val_data,
                              x_val_tr=val_data_tr,
                              x_val_te=val_data_te_ratings,
                              mapper=am_val
                              )
    print("Took {} seconds for training.".format(t))

    model_with_anneal.display_metrics()
    ndcg_val_with_anneal = model_with_anneal.ndcg_per_epoch()

    # Get optimal beta ################################################################################################
    optimal_beta = model_with_anneal.get_optimal_beta()
    print("The optimal beta is: ", optimal_beta)

    model_optimal_beta = Mult_VAE(n_users=train_data.shape[0],  # Number of unique users in the training set
                                  original_dim=train_data.shape[1],  # Number of unique items in the training set
                                  intermediate_dim=INTERMEDIATE_DIM,
                                  latent_dim=LATENT_DIM,
                                  n_epochs=EPOCHS,
                                  batch_size=BATCH_SIZE,
                                  k=TOP_K,
                                  verbose=0,
                                  seed=SEED,
                                  save_path=WEIGHTS_PATH_OPTIMAL_BETA,
                                  drop_encoder=0.5,
                                  drop_decoder=0.5,
                                  annealing=True,
                                  anneal_cap=optimal_beta,
                                  )

    with Timer() as t:
        model_optimal_beta.fit(x_train=train_data,
                               x_valid=val_data,
                               x_val_tr=val_data_tr,
                               x_val_te=val_data_te_ratings,
                               mapper=am_val
                               )

    print("Took {} seconds for training.".format(t))
    model_optimal_beta.display_metrics()
    ndcg_val_optimal_beta = model_optimal_beta.ndcg_per_epoch()

    # Use k = 10 prediction ############################################################################################
    with Timer() as t:
        # Model prediction on the training part of test set
        top_k = model_optimal_beta.recommend_k_items(x=test_data_tr,
                                                     k=10,
                                                     remove_seen=True
                                                     )

        # Convert sparse matrix back to df
        top_k_df = am_test.map_back_sparse(top_k, kind='prediction')
        test_df = am_test.map_back_sparse(test_data_te_ratings,
                                          kind='ratings')  # use test_data_te_, with the original ratings

    print("Took {} seconds for prediction.".format(t))

    # Use the ranking metrics for evaluation
    eval_map_3 = map_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_ndcg_3 = ndcg_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_precision_3 = precision_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_recall_3 = recall_at_k(test_df, top_k_df, col_prediction='prediction', k=10)

    print("MAP@10:\t\t%f" % eval_map_3,
          "NDCG@10:\t%f" % eval_ndcg_3,
          "Precision@10:\t%f" % eval_precision_3,
          "Recall@10: \t%f" % eval_recall_3, sep='\n')

    # Use k = 100s
    with Timer() as t:
        # Model prediction on the training part of test set
        top_k = model_optimal_beta.recommend_k_items(x=test_data_tr,
                                                     k=TOP_K,
                                                     remove_seen=True
                                                     )

        # Convert sparse matrix back to df
        top_k_df = am_test.map_back_sparse(top_k, kind='prediction')
        test_df = am_test.map_back_sparse(test_data_te_ratings,
                                          kind='ratings')  # use test_data_te_, with the original ratings

    print("Took {} seconds for prediction.".format(t))

    # Use the ranking metrics for evaluation
    eval_map_4 = map_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_ndcg_4 = ndcg_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_precision_4 = precision_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_recall_4 = recall_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)

    print("MAP@100:\t%f" % eval_map_4,
          "NDCG@100:\t%f" % eval_ndcg_4,
          "Precision@100:\t%f" % eval_precision_4,
          "Recall@100: \t%f" % eval_recall_4, sep='\n')

    # Plot setup
    plt.figure(figsize=(15, 5))
    sns.set(style='whitegrid')

    # Plot NDCG@k of validation sets for three models
    plt.plot(ndcg_val_without_anneal, color='b', linestyle='-', label='without anneal')
    plt.plot(ndcg_val_with_anneal, color='g', linestyle='-', label='with anneal at β=1')
    plt.plot(ndcg_val_optimal_beta, color='r', linestyle='-', label='with anneal at optimal β')

    # Add plot title and axis names
    plt.title('VALIDATION NDCG@100 FOR DIFFERENT MODELS \n', size=16)
    plt.xlabel('Epochs', size=14)
    plt.ylabel('NDCG@100', size=14)
    plt.legend(loc='lower right')

    plt.show()

    print("########## END OF MODEL TRAINING ##########")



def run_model_predict():

    # 경고메세지 끄기
    # warnings.filterwarnings(action='ignore')
    # 다시 출력하게 하기
    warnings.filterwarnings(action='default')


    # top k items to recommend
    TOP_K = 10
    # Select MovieLens data size: 100k, 1m, 10m, or 20m
    MOVIELENS_DATA_SIZE = '1m'
    # Model parameters
    HELDOUT_USERS = 600  # CHANGE FOR DIFFERENT DATASIZE
    INTERMEDIATE_DIM = 200
    LATENT_DIM = 70
    EPOCHS = 400
    EPOCHS = 1
    BATCH_SIZE = 100
    # temporary Path to save the optimal model's weights
    # tmp_dir = TemporaryDirectory()

    tmp_dir = f"{ENV.DATA_PATH}/m08"
    WEIGHTS_PATH_WITHOUT_ANNEAL = f"{tmp_dir}/mvae_weights_without_anneal.hdf5"
    WEIGHTS_PATH_WITH_ANNEAL = f"{tmp_dir}/mvae_weights_with_anneal.hdf5"
    WEIGHTS_PATH_OPTIMAL_BETA = f"{tmp_dir}/mvae_weights_optimal_beta.hdf5"

    print(tmp_dir)
    print(WEIGHTS_PATH_WITHOUT_ANNEAL)
    print(WEIGHTS_PATH_WITH_ANNEAL)
    print(WEIGHTS_PATH_OPTIMAL_BETA)

    SEED = 98765

    if os.path.isfile(f"{ENV.DATA_PATH}/m08/movielens.pickle"):
        df = pd.read_pickle(f"{ENV.DATA_PATH}/m08/movielens.pickle")
    else:
        df = movielens.load_pandas_df(
            size=MOVIELENS_DATA_SIZE,
            header=["userID", "itemID", "rating", "timestamp"]
        )
        df.to_pickle(f"{ENV.DATA_PATH}/m08/movielens.pickle")
    # df.head()
    # print(df.shape)

    # Binarize the data (only keep ratings >= 4)
    df_preferred = df[df['rating'] > 3.5]
    print(df_preferred.shape)
    df_low_rating = df[df['rating'] <= 3.5]
    # df.head()
    print(df_preferred.head(10))

    # Keep users who clicked on at least 5 movies
    df = min_rating_filter_pandas(df_preferred, min_rating=5, filter_by="user")
    # Keep movies that were clicked on by at least on 1 user
    df = min_rating_filter_pandas(df, min_rating=1, filter_by="item")
    # Obtain both usercount and itemcount after filtering
    usercount = df[['userID']].groupby('userID', as_index=False).size()
    itemcount = df[['itemID']].groupby('itemID', as_index=False).size()
    # Compute sparsity after filtering
    sparsity = 1. * df.shape[0] / (usercount.shape[0] * itemcount.shape[0])

    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (df.shape[0], usercount.shape[0], itemcount.shape[0], sparsity * 100))

    unique_users = sorted(df.userID.unique())
    np.random.seed(SEED)
    unique_users = np.random.permutation(unique_users)

    # Create train/validation/test users
    n_users = len(unique_users)
    print("Number of unique users:", n_users)
    train_users = unique_users[:(n_users - HELDOUT_USERS * 2)]
    print("Number of training users:", len(train_users))
    val_users = unique_users[(n_users - HELDOUT_USERS * 2): (n_users - HELDOUT_USERS)]
    print("Number of validation users:", len(val_users))
    test_users = unique_users[(n_users - HELDOUT_USERS):]
    print("Number of test users:", len(test_users))
    # For training set keep only users that are in train_users list
    train_set = df.loc[df['userID'].isin(train_users)]
    print("Number of training observations: ", train_set.shape[0])
    # For validation set keep only users that are in val_users list
    val_set = df.loc[df['userID'].isin(val_users)]
    print("Number of validation observations: ", val_set.shape[0])
    # For test set keep only users that are in test_users list
    test_set = df.loc[df['userID'].isin(test_users)]
    print("Number of test observations: ", test_set.shape[0])
    # train_set/val_set/test_set contain user - movie interactions with rating 4 or 5
    # Obtain list of unique movies used in training set
    unique_train_items = pd.unique(train_set['itemID'])
    print("Number of unique movies that rated in training set", unique_train_items.size)
    # For validation set keep only movies that used in training set
    val_set = val_set.loc[val_set['itemID'].isin(unique_train_items)]
    print("Number of validation observations after filtering: ", val_set.shape[0])
    # For test set keep only movies that used in training set
    test_set = test_set.loc[test_set['itemID'].isin(unique_train_items)]
    print("Number of test observations after filtering: ", test_set.shape[0])

    # Instantiate the sparse matrix generation for train, validation and test sets
    # use list of unique items from training set for all sets
    am_train = AffinityMatrix(df=train_set, items_list=unique_train_items)
    am_val = AffinityMatrix(df=val_set, items_list=unique_train_items)
    am_test = AffinityMatrix(df=test_set, items_list=unique_train_items)

    # Obtain the sparse matrix for train, validation and test sets
    train_data, _, _ = am_train.gen_affinity_matrix()
    # print(train_data.shape)
    val_data, val_map_users, val_map_items = am_val.gen_affinity_matrix()
    # print(val_data.shape)
    test_data, test_map_users, test_map_items = am_test.gen_affinity_matrix()
    # print(test_data.shape)

    # Split validation and test data into training and testing parts
    val_data_tr, val_data_te = numpy_stratified_split(val_data, ratio=0.75, seed=SEED)
    test_data_tr, test_data_te = numpy_stratified_split(test_data, ratio=0.75, seed=SEED)

    # Binarize train, validation and test data
    train_data = binarize(a=train_data, threshold=3.5)
    val_data = binarize(a=val_data, threshold=3.5)
    test_data = binarize(a=test_data, threshold=3.5)

    # Binarize validation data: training part
    val_data_tr = binarize(a=val_data_tr, threshold=3.5)

    # Binarize validation data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
    val_data_te_ratings = val_data_te.copy()
    val_data_te = binarize(a=val_data_te, threshold=3.5)

    # Binarize test data: training part
    test_data_tr = binarize(a=test_data_tr, threshold=3.5)

    # Binarize test data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
    test_data_te_ratings = test_data_te.copy()
    test_data_te = binarize(a=test_data_te, threshold=3.5)
    print(test_data_te)

    # retrieve real ratings from initial dataset

    test_data_te_ratings = pd.DataFrame(test_data_te_ratings)
    val_data_te_ratings = pd.DataFrame(val_data_te_ratings)

    for index, i in df_low_rating.iterrows():
        user_old = i['userID']  # old value
        item_old = i['itemID']  # old value

        if (test_map_users.get(user_old) is not None) and (test_map_items.get(item_old) is not None):
            user_new = test_map_users.get(user_old)  # new value
            item_new = test_map_items.get(item_old)  # new value
            rating = i['rating']
            test_data_te_ratings.at[user_new, item_new] = rating

        if (val_map_users.get(user_old) is not None) and (val_map_items.get(item_old) is not None):
            user_new = val_map_users.get(user_old)  # new value
            item_new = val_map_items.get(item_old)  # new value
            rating = i['rating']
            val_data_te_ratings.at[user_new, item_new] = rating

    val_data_te_ratings = val_data_te_ratings.to_numpy()
    test_data_te_ratings = test_data_te_ratings.to_numpy()
    # test_data_te_ratings

    # Just checking
    print(np.sum(val_data))
    print(np.sum(val_data_tr))
    print(np.sum(val_data_te))

    # Just checking
    print(np.sum(test_data))
    print(np.sum(test_data_tr))
    print(np.sum(test_data_te))

    ####################################################################################################################
    model_without_anneal = Mult_VAE(n_users=train_data.shape[0],  # Number of unique users in the training set
                                    original_dim=train_data.shape[1],  # Number of unique items in the training set
                                    intermediate_dim=INTERMEDIATE_DIM,
                                    latent_dim=LATENT_DIM,
                                    n_epochs=EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    k=TOP_K,
                                    verbose=0,
                                    seed=SEED,
                                    save_path=WEIGHTS_PATH_WITHOUT_ANNEAL,
                                    drop_encoder=0.5,
                                    drop_decoder=0.5,
                                    annealing=False,
                                    beta=1.0
                                    )

    # Loading Trained Data File ########################################################################################

    # Use k = 10
    with Timer() as t:
        # Model prediction on the training part of test set
        top_k = model_without_anneal.recommend_k_items(x=test_data_tr,
                                                       k=10,
                                                       remove_seen=True
                                                       )
        print("############prediction#############################")
        print(top_k)
        print(test_data_tr.shape)
        print(test_data)
        # Convert sparse matrix back to df
        top_k_df = am_test.map_back_sparse(top_k, kind='prediction')
        test_df = am_test.map_back_sparse(test_data_te_ratings,
                                          kind='ratings')  # use test_data_te_, with the original ratings

    print("Took {} seconds for prediction.".format(t))

    # Use the ranking metrics for evaluation ###########################################################################
    eval_map_1 = map_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_ndcg_1 = ndcg_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_precision_1 = precision_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_recall_1 = recall_at_k(test_df, top_k_df, col_prediction='prediction', k=10)

    print("MAP@10:\t\t%f" % eval_map_1,
          "NDCG@10:\t%f" % eval_ndcg_1,
          "Precision@10:\t%f" % eval_precision_1,
          "Recall@10: \t%f" % eval_recall_1, sep='\n')
    ####################################################################################################################

    # Use k = TOP_K
    with Timer() as t:
        # Model prediction on the training part of test set
        top_k = model_without_anneal.recommend_k_items(x=test_data_tr,
                                                       k=TOP_K,
                                                       remove_seen=True
                                                       )
        # Convert sparse matrix back to df
        top_k_df = am_test.map_back_sparse(top_k, kind='prediction')
        test_df = am_test.map_back_sparse(test_data_te_ratings,
                                          kind='ratings')  # use test_data_te_, with the original ratings

    print("Took {} seconds for prediction.".format(t))

    # Use the ranking metrics for evaluation ###########################################################################
    eval_map_2 = map_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_ndcg_2 = ndcg_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_precision_2 = precision_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_recall_2 = recall_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    print("MAP@100:\t%f" % eval_map_2,
          "NDCG@100:\t%f" % eval_ndcg_2,
          "Precision@100:\t%f" % eval_precision_2,
          "Recall@100: \t%f" % eval_recall_2, sep='\n')
    ####################################################################################################################
    # model -2
    model_with_anneal = Mult_VAE(n_users=train_data.shape[0],  # Number of unique users in the training set
                                 original_dim=train_data.shape[1],  # Number of unique items in the training set
                                 intermediate_dim=INTERMEDIATE_DIM,
                                 latent_dim=LATENT_DIM,
                                 n_epochs=EPOCHS,
                                 batch_size=BATCH_SIZE,
                                 k=TOP_K,
                                 verbose=0,
                                 seed=SEED,
                                 save_path=WEIGHTS_PATH_WITH_ANNEAL,
                                 drop_encoder=0.5,
                                 drop_decoder=0.5,
                                 annealing=True,
                                 anneal_cap=1.0,
                                 )


    # model_with_anneal.display_metrics()
    # ndcg_val_with_anneal = model_with_anneal.ndcg_per_epoch()

    # Get optimal beta ################################################################################################
    # optimal_beta = model_with_anneal.get_optimal_beta()
    optimal_beta = 1
    print("The optimal beta is: ", optimal_beta)

    model_optimal_beta = Mult_VAE(n_users=train_data.shape[0],  # Number of unique users in the training set
                                  original_dim=train_data.shape[1],  # Number of unique items in the training set
                                  intermediate_dim=INTERMEDIATE_DIM,
                                  latent_dim=LATENT_DIM,
                                  n_epochs=EPOCHS,
                                  batch_size=BATCH_SIZE,
                                  k=TOP_K,
                                  verbose=0,
                                  seed=SEED,
                                  save_path=WEIGHTS_PATH_OPTIMAL_BETA,
                                  drop_encoder=0.5,
                                  drop_decoder=0.5,
                                  annealing=True,
                                  anneal_cap=optimal_beta,
                                  )


    # model_optimal_beta.display_metrics()
    # ndcg_val_optimal_beta = model_optimal_beta.ndcg_per_epoch()

    # Use k = 10 prediction ############################################################################################
    with Timer() as t:
        # Model prediction on the training part of test set
        top_k = model_optimal_beta.recommend_k_items(x=test_data_tr,
                                                     k=10,
                                                     remove_seen=True
                                                     )
        print("############prediction#############################")
        print(top_k)
        # Convert sparse matrix back to df
        top_k_df = am_test.map_back_sparse(top_k, kind='prediction')
        test_df = am_test.map_back_sparse(test_data_te_ratings,
                                          kind='ratings')  # use test_data_te_, with the original ratings

    print("Took {} seconds for prediction.".format(t))

    # Use the ranking metrics for evaluation
    eval_map_3 = map_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_ndcg_3 = ndcg_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_precision_3 = precision_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_recall_3 = recall_at_k(test_df, top_k_df, col_prediction='prediction', k=10)

    print("MAP@10:\t\t%f" % eval_map_3,
          "NDCG@10:\t%f" % eval_ndcg_3,
          "Precision@10:\t%f" % eval_precision_3,
          "Recall@10: \t%f" % eval_recall_3, sep='\n')

    # Use k = 100s
    with Timer() as t:
        # Model prediction on the training part of test set
        top_k = model_optimal_beta.recommend_k_items(x=test_data_tr,
                                                     k=TOP_K,
                                                     remove_seen=True
                                                     )
        print("############prediction#############################")
        print(top_k)

        # Convert sparse matrix back to df
        top_k_df = am_test.map_back_sparse(top_k, kind='prediction')
        test_df = am_test.map_back_sparse(test_data_te_ratings,
                                          kind='ratings')  # use test_data_te_, with the original ratings

    print("Took {} seconds for prediction.".format(t))


    # Use the ranking metrics for evaluation
    eval_map_4 = map_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_ndcg_4 = ndcg_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_precision_4 = precision_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_recall_4 = recall_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)

    print("MAP@100:\t%f" % eval_map_4,
          "NDCG@100:\t%f" % eval_ndcg_4,
          "Precision@100:\t%f" % eval_precision_4,
          "Recall@100: \t%f" % eval_recall_4, sep='\n')

    # Plot setup
    plt.figure(figsize=(15, 5))
    sns.set(style='whitegrid')

    # Plot NDCG@k of validation sets for three models
    #plt.plot(ndcg_val_without_anneal, color='b', linestyle='-', label='without anneal')
    #plt.plot(ndcg_val_with_anneal, color='g', linestyle='-', label='with anneal at β=1')
    #plt.plot(ndcg_val_optimal_beta, color='r', linestyle='-', label='with anneal at optimal β')

    # Add plot title and axis names
    plt.title('VALIDATION NDCG@100 FOR DIFFERENT MODELS \n', size=16)
    plt.xlabel('Epochs', size=14)
    plt.ylabel('NDCG@100', size=14)
    plt.legend(loc='lower right')

    plt.show()

    print("########## END OF MODEL TRAINING ##########")



def run_model_train():
    # 경고메세지 끄기
    warnings.filterwarnings(action='ignore')
    # 다시 출력하게 하기
    # warnings.filterwarnings(action='default')

    print(f"System version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Tensorflow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")

    # top k items to recommend
    TOP_K = 10
    # Select MovieLens data size: 100k, 1m, 10m, or 20m
    MOVIELENS_DATA_SIZE = '1m'
    # Model parameters
    HELDOUT_USERS = 600  # CHANGE FOR DIFFERENT DATASIZE
    INTERMEDIATE_DIM = 200
    LATENT_DIM = 70
    EPOCHS = 400
    EPOCHS = 200
    BATCH_SIZE = 100
    # temporary Path to save the optimal model's weights
    # tmp_dir = TemporaryDirectory()

    tmp_dir = f"{ENV.DATA_PATH}/m08"
    WEIGHTS_PATH_WITHOUT_ANNEAL = f"{tmp_dir}/mvae_weights_without_anneal.hdf5"
    WEIGHTS_PATH_WITH_ANNEAL = f"{tmp_dir}/mvae_weights_with_anneal.hdf5"
    WEIGHTS_PATH_OPTIMAL_BETA = f"{tmp_dir}/mvae_weights_optimal_beta.hdf5"

    print(tmp_dir)
    print(WEIGHTS_PATH_WITHOUT_ANNEAL)
    print(WEIGHTS_PATH_WITH_ANNEAL)
    print(WEIGHTS_PATH_OPTIMAL_BETA)

    SEED = 98765

    if os.path.isfile(f"{ENV.DATA_PATH}/m08/movielens.pickle"):
        df = pd.read_pickle(f"{ENV.DATA_PATH}/m08/movielens.pickle")
    else:
        df = movielens.load_pandas_df(
            size=MOVIELENS_DATA_SIZE,
            header=["userID", "itemID", "rating", "timestamp"]
        )
        df.to_pickle(f"{ENV.DATA_PATH}/m08/movielens.pickle")
    # df.head()
    # print(df.shape)

    # Binarize the data (only keep ratings >= 4)
    df_preferred = df[df['rating'] > 3.5]
    print(df_preferred.shape)
    df_low_rating = df[df['rating'] <= 3.5]
    # df.head()
    print(df_preferred.head(10))

    # Keep users who clicked on at least 5 movies
    df = min_rating_filter_pandas(df_preferred, min_rating=5, filter_by="user")
    # Keep movies that were clicked on by at least on 1 user
    df = min_rating_filter_pandas(df, min_rating=1, filter_by="item")
    # Obtain both usercount and itemcount after filtering
    usercount = df[['userID']].groupby('userID', as_index=False).size()
    itemcount = df[['itemID']].groupby('itemID', as_index=False).size()
    # Compute sparsity after filtering
    sparsity = 1. * df.shape[0] / (usercount.shape[0] * itemcount.shape[0])

    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (df.shape[0], usercount.shape[0], itemcount.shape[0], sparsity * 100))

    unique_users = sorted(df.userID.unique())
    np.random.seed(SEED)
    unique_users = np.random.permutation(unique_users)

    # Create train/validation/test users
    n_users = len(unique_users)
    print("Number of unique users:", n_users)
    train_users = unique_users[:(n_users - HELDOUT_USERS * 2)]
    print("Number of training users:", len(train_users))
    val_users = unique_users[(n_users - HELDOUT_USERS * 2): (n_users - HELDOUT_USERS)]
    print("Number of validation users:", len(val_users))
    test_users = unique_users[(n_users - HELDOUT_USERS):]
    print("Number of test users:", len(test_users))
    # For training set keep only users that are in train_users list
    train_set = df.loc[df['userID'].isin(train_users)]
    print("Number of training observations: ", train_set.shape[0])
    # For validation set keep only users that are in val_users list
    val_set = df.loc[df['userID'].isin(val_users)]
    print("Number of validation observations: ", val_set.shape[0])
    # For test set keep only users that are in test_users list
    test_set = df.loc[df['userID'].isin(test_users)]
    print("Number of test observations: ", test_set.shape[0])
    # train_set/val_set/test_set contain user - movie interactions with rating 4 or 5
    # Obtain list of unique movies used in training set
    unique_train_items = pd.unique(train_set['itemID'])
    print("Number of unique movies that rated in training set", unique_train_items.size)
    # For validation set keep only movies that used in training set
    val_set = val_set.loc[val_set['itemID'].isin(unique_train_items)]
    print("Number of validation observations after filtering: ", val_set.shape[0])
    # For test set keep only movies that used in training set
    test_set = test_set.loc[test_set['itemID'].isin(unique_train_items)]
    print("Number of test observations after filtering: ", test_set.shape[0])

    # Instantiate the sparse matrix generation for train, validation and test sets
    # use list of unique items from training set for all sets
    am_train = AffinityMatrix(df=train_set, items_list=unique_train_items)
    am_val = AffinityMatrix(df=val_set, items_list=unique_train_items)
    am_test = AffinityMatrix(df=test_set, items_list=unique_train_items)

    # Obtain the sparse matrix for train, validation and test sets
    train_data, _, _ = am_train.gen_affinity_matrix()
    # print(train_data.shape)
    val_data, val_map_users, val_map_items = am_val.gen_affinity_matrix()
    # print(val_data.shape)
    test_data, test_map_users, test_map_items = am_test.gen_affinity_matrix()
    # print(test_data.shape)

    # Split validation and test data into training and testing parts
    val_data_tr, val_data_te = numpy_stratified_split(val_data, ratio=0.75, seed=SEED)
    test_data_tr, test_data_te = numpy_stratified_split(test_data, ratio=0.75, seed=SEED)

    # Binarize train, validation and test data
    train_data = binarize(a=train_data, threshold=3.5)
    val_data = binarize(a=val_data, threshold=3.5)
    test_data = binarize(a=test_data, threshold=3.5)

    # Binarize validation data: training part
    val_data_tr = binarize(a=val_data_tr, threshold=3.5)

    # Binarize validation data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
    val_data_te_ratings = val_data_te.copy()
    val_data_te = binarize(a=val_data_te, threshold=3.5)

    # Binarize test data: training part
    test_data_tr = binarize(a=test_data_tr, threshold=3.5)

    # Binarize test data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
    test_data_te_ratings = test_data_te.copy()
    test_data_te = binarize(a=test_data_te, threshold=3.5)

    # retrieve real ratings from initial dataset

    test_data_te_ratings = pd.DataFrame(test_data_te_ratings)
    val_data_te_ratings = pd.DataFrame(val_data_te_ratings)

    for index, i in df_low_rating.iterrows():
        user_old = i['userID']  # old value
        item_old = i['itemID']  # old value

        if (test_map_users.get(user_old) is not None) and (test_map_items.get(item_old) is not None):
            user_new = test_map_users.get(user_old)  # new value
            item_new = test_map_items.get(item_old)  # new value
            rating = i['rating']
            test_data_te_ratings.at[user_new, item_new] = rating

        if (val_map_users.get(user_old) is not None) and (val_map_items.get(item_old) is not None):
            user_new = val_map_users.get(user_old)  # new value
            item_new = val_map_items.get(item_old)  # new value
            rating = i['rating']
            val_data_te_ratings.at[user_new, item_new] = rating

    val_data_te_ratings = val_data_te_ratings.to_numpy()
    test_data_te_ratings = test_data_te_ratings.to_numpy()
    # test_data_te_ratings

    # Just checking
    print(np.sum(val_data))
    print(np.sum(val_data_tr))
    print(np.sum(val_data_te))

    # Just checking
    print(np.sum(test_data))
    print(np.sum(test_data_tr))
    print(np.sum(test_data_te))

    ####################################################################################################################
    model_without_anneal = Mult_VAE(n_users=train_data.shape[0],  # Number of unique users in the training set
                                    original_dim=train_data.shape[1],  # Number of unique items in the training set
                                    intermediate_dim=INTERMEDIATE_DIM,
                                    latent_dim=LATENT_DIM,
                                    n_epochs=EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    k=TOP_K,
                                    verbose=0,
                                    seed=SEED,
                                    save_path=WEIGHTS_PATH_WITHOUT_ANNEAL,
                                    drop_encoder=0.5,
                                    drop_decoder=0.5,
                                    annealing=False,
                                    beta=1.0
                                    )

    # traing ###########################################################################################################
    with Timer() as t:
        model_without_anneal.fit(x_train=train_data,
                                 x_valid=val_data,
                                 x_val_tr=val_data_tr,
                                 x_val_te=val_data_te_ratings,
                                 mapper=am_val
                                 )
    print("Took {} seconds for training.".format(t))

    model_without_anneal.display_metrics()
    ndcg_val_without_anneal = model_without_anneal.ndcg_per_epoch()

    # Use k = 10 #######################################################################################################
    with Timer() as t:
        # Model prediction on the training part of test set
        top_k = model_without_anneal.recommend_k_items(x=test_data_tr,
                                                       k=10,
                                                       remove_seen=True
                                                       )
        print("############prediction#############################")
        print(top_k)
        # Convert sparse matrix back to df
        top_k_df = am_test.map_back_sparse(top_k, kind='prediction')
        test_df = am_test.map_back_sparse(test_data_te_ratings,
                                          kind='ratings')  # use test_data_te_, with the original ratings

    print("Took {} seconds for prediction.".format(t))
    # Use the ranking metrics for evaluation ###########################################################################
    eval_map_1 = map_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_ndcg_1 = ndcg_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_precision_1 = precision_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_recall_1 = recall_at_k(test_df, top_k_df, col_prediction='prediction', k=10)

    print("MAP@10:\t\t%f" % eval_map_1,
          "NDCG@10:\t%f" % eval_ndcg_1,
          "Precision@10:\t%f" % eval_precision_1,
          "Recall@10: \t%f" % eval_recall_1, sep='\n')

    ####################################################################################################################
    # Use k = TOP_K
    with Timer() as t:
        # Model prediction on the training part of test set
        top_k = model_without_anneal.recommend_k_items(x=test_data_tr,
                                                       k=TOP_K,
                                                       remove_seen=True
                                                       )
        print("############prediction#############################")
        print(top_k)
        # Convert sparse matrix back to df
        top_k_df = am_test.map_back_sparse(top_k, kind='prediction')
        test_df = am_test.map_back_sparse(test_data_te_ratings,
                                          kind='ratings')  # use test_data_te_, with the original ratings

    print("Took {} seconds for prediction.".format(t))
    # Use the ranking metrics for evaluation ###########################################################################
    eval_map_2 = map_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_ndcg_2 = ndcg_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_precision_2 = precision_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_recall_2 = recall_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    print("MAP@100:\t%f" % eval_map_2,
          "NDCG@100:\t%f" % eval_ndcg_2,
          "Precision@100:\t%f" % eval_precision_2,
          "Recall@100: \t%f" % eval_recall_2, sep='\n')

    ####################################################################################################################
    # model -2
    model_with_anneal = Mult_VAE(n_users=train_data.shape[0],  # Number of unique users in the training set
                                 original_dim=train_data.shape[1],  # Number of unique items in the training set
                                 intermediate_dim=INTERMEDIATE_DIM,
                                 latent_dim=LATENT_DIM,
                                 n_epochs=EPOCHS,
                                 batch_size=BATCH_SIZE,
                                 k=TOP_K,
                                 verbose=0,
                                 seed=SEED,
                                 save_path=WEIGHTS_PATH_WITH_ANNEAL,
                                 drop_encoder=0.5,
                                 drop_decoder=0.5,
                                 annealing=True,
                                 anneal_cap=1.0,
                                 )
    # traing
    with Timer() as t:
        model_with_anneal.fit(x_train=train_data,
                              x_valid=val_data,
                              x_val_tr=val_data_tr,
                              x_val_te=val_data_te_ratings,
                              mapper=am_val
                              )
    print("Took {} seconds for training.".format(t))
    model_with_anneal.display_metrics()
    ndcg_val_with_anneal = model_with_anneal.ndcg_per_epoch()
    # Get optimal beta ################################################################################################
    optimal_beta = model_with_anneal.get_optimal_beta()
    print("The optimal beta is: ", optimal_beta)

    model_optimal_beta = Mult_VAE(n_users=train_data.shape[0],  # Number of unique users in the training set
                                  original_dim=train_data.shape[1],  # Number of unique items in the training set
                                  intermediate_dim=INTERMEDIATE_DIM,
                                  latent_dim=LATENT_DIM,
                                  n_epochs=EPOCHS,
                                  batch_size=BATCH_SIZE,
                                  k=TOP_K,
                                  verbose=0,
                                  seed=SEED,
                                  save_path=WEIGHTS_PATH_OPTIMAL_BETA,
                                  drop_encoder=0.5,
                                  drop_decoder=0.5,
                                  annealing=True,
                                  anneal_cap=optimal_beta,
                                  )

    with Timer() as t:
        model_optimal_beta.fit(x_train=train_data,
                               x_valid=val_data,
                               x_val_tr=val_data_tr,
                               x_val_te=val_data_te_ratings,
                               mapper=am_val
                               )

    print("Took {} seconds for training.".format(t))

    model_optimal_beta.display_metrics()
    ndcg_val_optimal_beta = model_optimal_beta.ndcg_per_epoch()

    # Use k = 10 prediction ############################################################################################
    with Timer() as t:
        # Model prediction on the training part of test set
        top_k = model_optimal_beta.recommend_k_items(x=test_data_tr,
                                                     k=10,
                                                     remove_seen=True
                                                     )

        # Convert sparse matrix back to df
        top_k_df = am_test.map_back_sparse(top_k, kind='prediction')
        test_df = am_test.map_back_sparse(test_data_te_ratings,
                                          kind='ratings')  # use test_data_te_, with the original ratings

    print("Took {} seconds for prediction.".format(t))

    # Use the ranking metrics for evaluation
    eval_map_3 = map_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_ndcg_3 = ndcg_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_precision_3 = precision_at_k(test_df, top_k_df, col_prediction='prediction', k=10)
    eval_recall_3 = recall_at_k(test_df, top_k_df, col_prediction='prediction', k=10)

    print("MAP@10:\t\t%f" % eval_map_3,
          "NDCG@10:\t%f" % eval_ndcg_3,
          "Precision@10:\t%f" % eval_precision_3,
          "Recall@10: \t%f" % eval_recall_3, sep='\n')

    # Use k = 100s
    with Timer() as t:
        # Model prediction on the training part of test set
        top_k = model_optimal_beta.recommend_k_items(x=test_data_tr,
                                                     k=TOP_K,
                                                     remove_seen=True
                                                     )

        # Convert sparse matrix back to df
        top_k_df = am_test.map_back_sparse(top_k, kind='prediction')
        test_df = am_test.map_back_sparse(test_data_te_ratings,
                                          kind='ratings')  # use test_data_te_, with the original ratings

    print("Took {} seconds for prediction.".format(t))

    # Use the ranking metrics for evaluation
    eval_map_4 = map_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_ndcg_4 = ndcg_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_precision_4 = precision_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)
    eval_recall_4 = recall_at_k(test_df, top_k_df, col_prediction='prediction', k=TOP_K)

    print("MAP@100:\t%f" % eval_map_4,
          "NDCG@100:\t%f" % eval_ndcg_4,
          "Precision@100:\t%f" % eval_precision_4,
          "Recall@100: \t%f" % eval_recall_4, sep='\n')

    # Plot setup
    plt.figure(figsize=(15, 5))
    sns.set(style='whitegrid')

    # Plot NDCG@k of validation sets for three models
    plt.plot(ndcg_val_without_anneal, color='b', linestyle='-', label='without anneal')
    plt.plot(ndcg_val_with_anneal, color='g', linestyle='-', label='with anneal at β=1')
    plt.plot(ndcg_val_optimal_beta, color='r', linestyle='-', label='with anneal at optimal β')

    # Add plot title and axis names
    plt.title('VALIDATION NDCG@100 FOR DIFFERENT MODELS \n', size=16)
    plt.xlabel('Epochs', size=14)
    plt.ylabel('NDCG@100', size=14)
    plt.legend(loc='lower right')

    plt.show()

    print("########## END OF MODEL TRAINING ##########")
