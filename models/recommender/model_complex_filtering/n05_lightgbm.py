import os
import sys
import numpy as np
import lightgbm as lgb
import pandas as pd
import category_encoders as ce

from tempfile import TemporaryDirectory
from sklearn.metrics import roc_auc_score, log_loss

import recommenders.datasets.criteo as criteo
import recommenders.models.lightgbm.lightgbm_utils as lgbm_utils
from recommenders.utils.notebook_utils import store_metadata

MAX_LEAF = 64
MIN_DATA = 20
NUM_OF_TREES = 100
TREE_LEARNING_RATE = 0.15
EARLY_STOPPING_ROUNDS = 20
METRIC = "auc"
SIZE = "sample"

params = {
    "task": "train",
    "boosting_type": "gbdt",
    "num_class": 1,
    "objective": "binary",
    "metric": METRIC,
    "num_leaves": MAX_LEAF,
    "min_data": MIN_DATA,
    "boost_from_average": True,
    # set it according to your cpu cores.
    "num_threads": 20,
    "feature_fraction": 0.8,
    "learning_rate": TREE_LEARNING_RATE,
}

nume_cols = ["I" + str(i) for i in range(1, 14)]
cate_cols = ["C" + str(i) for i in range(1, 27)]
label_col = "Label"

# data load - how fix it? for time
header = [label_col] + nume_cols + cate_cols
with TemporaryDirectory() as tmp:
    all_data = criteo.load_pandas_df(size=SIZE, local_cache_path=tmp, header=header)

print(all_data.head())
all_data.head