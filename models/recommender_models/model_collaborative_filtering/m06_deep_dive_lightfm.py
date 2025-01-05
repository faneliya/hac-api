import os
import sys
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightfm
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm import cross_validation
from lightfm.evaluation import precision_at_k as lightfm_prec_at_k
from lightfm.evaluation import recall_at_k as lightfm_recall_at_k

from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k
from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.models.lightfm.lightfm_utils import (
    track_model_metrics,
    prepare_test_df,
    prepare_all_predictions,
    compare_metric,
    similar_users,
    similar_items,
)
from recommenders.utils.notebook_utils import store_metadata
import util.constant as ENV
import warnings

def run_model():

    print("m06_deep_dive_lightfm started")
    print("System version: {}".format(sys.version))
    print("LightFM version: {}".format(lightfm.__version__))
    # Select MovieLens data size

    MOVIELENS_DATA_SIZE = '100k'

    # default number of recommendations
    K = 10
    # percentage of data used for testing
    TEST_PERCENTAGE = 0.25
    # model learning rate
    LEARNING_RATE = 0.25
    # no of latent factors
    NO_COMPONENTS = 20
    # no of epochs to fit model
    NO_EPOCHS = 20
    # no of threads to fit model
    NO_THREADS = 32
    # regularisation for both user and item features
    ITEM_ALPHA = 1e-6
    USER_ALPHA = 1e-6

    # seed for pseudonumber generations
    SEED = 42

    if os.path.isfile(f"{ENV.DATA_PATH}/m06/movielens.pickle"):
        data = pd.read_pickle(f"{ENV.DATA_PATH}/m06/movielens.pickle")
    else:
        data = movielens.load_pandas_df(
            size=MOVIELENS_DATA_SIZE,
            genres_col='genre',
            header=["userID", "itemID", "rating"]
        )
        data.to_pickle(f"{ENV.DATA_PATH}/m06/movielens.pickle")

    '''
    data = movielens.load_pandas_df(
        size=MOVIELENS_DATA_SIZE,
        genres_col='genre',
        header=["userID", "itemID", "rating"]
    )
    '''

    # quick look at the data
    data.sample(5, random_state=SEED)
    dataset = Dataset()
    dataset.fit(users=data['userID'],
                items=data['itemID'])

    # quick check to determine the number of unique users and items in the data
    num_users, num_topics = dataset.interactions_shape()
    print(f'Num users: {num_users}, num_topics: {num_topics}.')

    (interactions, weights) = dataset.build_interactions(data.iloc[:, 0:3].values)

    train_interactions, test_interactions = cross_validation.random_train_test_split(
        interactions, test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED))

    print(f"Shape of train interactions: {train_interactions.shape}")
    print(f"Shape of test interactions: {test_interactions.shape}")

    ####################################################################################################################
    model1 = LightFM(loss='warp', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     random_state=np.random.RandomState(SEED))

    model1.fit(interactions=train_interactions,
              epochs=NO_EPOCHS);

    uids, iids, interaction_data = cross_validation._shuffle(
        interactions.row, interactions.col, interactions.data,
        random_state=np.random.RandomState(SEED))

    cutoff = int((1.0 - TEST_PERCENTAGE) * len(uids))
    test_idx = slice(cutoff, None)

    uid_map, ufeature_map, iid_map, ifeature_map = dataset.mapping()

    with Timer() as test_time:
        test_df = prepare_test_df(test_idx, uids, iids, uid_map, iid_map, weights)

    print(f"Took {test_time.interval:.1f} seconds for prepare and predict test data.")

    time_reco1 = test_time.interval
    test_df.sample(5, random_state=SEED)

    with Timer() as test_time:
        all_predictions = prepare_all_predictions(data, uid_map, iid_map,
                                                  interactions=train_interactions,
                                                  model=model1,
                                                  num_threads=NO_THREADS)

    print(f"Took {test_time.interval:.1f} seconds for prepare and predict all data.")
    time_reco2 = test_time.interval

    all_predictions.sample(5, random_state=SEED)

    with Timer() as test_time:
        eval_precision = precision_at_k(rating_true=test_df,
                                        rating_pred=all_predictions, k=K)
        eval_recall = recall_at_k(test_df, all_predictions, k=K)

    time_reco3 = test_time.interval

    with Timer() as test_time:
        eval_precision_lfm = lightfm_prec_at_k(model1, test_interactions,
                                               train_interactions, k=K).mean()
        eval_recall_lfm = lightfm_recall_at_k(model1, test_interactions,
                                              train_interactions, k=K).mean()
    time_lfm = test_time.interval

    print(
        "------ Using Repo's evaluation methods ------",
        f"Precision@K:\t{eval_precision:.6f}",
        f"Recall@K:\t{eval_recall:.6f}",
        "\n------ Using LightFM evaluation methods ------",
        f"Precision@K:\t{eval_precision_lfm:.6f}",
        f"Recall@K:\t{eval_recall_lfm:.6f}",
        sep='\n')

    # split the genre based on the separator
    movie_genre = [x.split('|') for x in data['genre']]

    # retrieve the all the unique genres in the data
    all_movie_genre = sorted(list(set(itertools.chain.from_iterable(movie_genre))))
    # quick look at the all the genres within the data
    all_movie_genre

    user_feature_URL = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.user'
    columns = ['userID','age','gender','occupation','zipcode']
    user_data = pd.read_table(user_feature_URL, sep='|', header=None, names=columns)

    # merging user feature with existing data
    new_data = data.merge(user_data[['userID','occupation']], left_on='userID', right_on='userID')
    # quick look at the merged data
    new_data.sample(5, random_state=SEED)

    # retrieve all the unique occupations in the data
    all_occupations = sorted(list(set(new_data['occupation'])))

    dataset2 = Dataset()
    dataset2.fit(data['userID'],
                data['itemID'],
                item_features=all_movie_genre,
                user_features=all_occupations)

    item_features = dataset2.build_item_features((x, y) for x,y in zip(data.itemID, movie_genre))
    user_features = dataset2.build_user_features((x, [y]) for x,y in zip(new_data.userID, new_data['occupation']))
    interactions2, weights2 = dataset2.build_interactions(data.iloc[:, 0:3].values)

    train_interactions2, test_interactions2 = cross_validation.random_train_test_split(
        interactions2,
        test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED)
    )

    ####################################################################################################################
    model2 = LightFM(loss='warp', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     item_alpha=ITEM_ALPHA,
                     user_alpha=USER_ALPHA,
                     random_state=np.random.RandomState(SEED)
                    )

    model2.fit(interactions=train_interactions2,
               user_features=user_features,
               item_features=item_features,
               epochs=NO_EPOCHS
               )

    uids, iids, interaction_data = cross_validation._shuffle(
        interactions2.row,
        interactions2.col,
        interactions2.data,
        random_state=np.random.RandomState(SEED)
    )

    uid_map, ufeature_map, iid_map, ifeature_map = dataset2.mapping()

    with Timer() as test_time:
        test_df2 = prepare_test_df(test_idx, uids, iids, uid_map, iid_map, weights2)

    print(f"Took {test_time.interval:.1f} seconds for prepare and predict test data.")

    with Timer() as test_time:
        all_predictions2 = prepare_all_predictions(data, uid_map, iid_map,
                                                  interactions=train_interactions2,
                                                   user_features=user_features,
                                                   item_features=item_features,
                                                   model=model2,
                                                   num_threads=NO_THREADS)

    print(f"Took {test_time.interval:.1f} seconds for prepare and predict all data.")

    eval_precision2 = precision_at_k(rating_true=test_df2,
                                    rating_pred=all_predictions2, k=K)
    eval_recall2 = recall_at_k(test_df2, all_predictions2, k=K)

    print(
        "------ Using only explicit ratings ------",
        f"Precision@K:\t{eval_precision:.6f}",
        f"Recall@K:\t{eval_recall:.6f}",
        "\n------ Using both implicit and explicit ratings ------",
        f"Precision@K:\t{eval_precision2:.6f}",
        f"Recall@K:\t{eval_recall2:.6f}",
        sep='\n')

    print(
        "------ Using Repo's evaluation methods ------",
        f"Time [sec]:\t{(time_reco1+time_reco2+time_reco3):.1f}",
        "\n------ Using LightFM evaluation methods ------",
        f"Time [sec]:\t{time_lfm:.1f}",
        sep='\n')

    output1, _ = track_model_metrics(model=model1,
                                     train_interactions=train_interactions,
                                     test_interactions=test_interactions,
                                     k=K,
                                     no_epochs=NO_EPOCHS,
                                     no_threads=NO_THREADS)

    output2, _ = track_model_metrics(model=model2,
                                     train_interactions=train_interactions2,
                                     test_interactions=test_interactions2,
                                     k=K,
                                     no_epochs=NO_EPOCHS,
                                     no_threads=NO_THREADS,
                                     item_features=item_features,
                                     user_features=user_features)

    for i in ['Precision', 'Recall']:
        sns.set_palette("Set2")
        plt.figure()
        sns.scatterplot(x="epoch",
                        y="value",
                        hue='data',
                        data=compare_metric(df_list = [output1, output2], metric=i)
                       ).set_title(f'{i} comparison using test set');


    _, user_embeddings = model2.get_user_representations(features=user_features)

    user_embeddings

    similar_users(user_id=1,
                  user_features=user_features,
                  model=model2)

    _, item_embeddings = model2.get_item_representations(features=item_features)

    item_embeddings

    similar_items(item_id=10,
                  item_features=item_features,
                  model=model2)

    # Record results for tests - ignore this cell
    store_metadata("eval_precision", eval_precision)
    store_metadata("eval_recall", eval_recall)
    store_metadata("eval_precision2", eval_precision2)
    store_metadata("eval_recall2", eval_recall2)
