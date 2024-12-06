import os
import sys
from tempfile import TemporaryDirectory
import xlearn as xl
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import seaborn as sns

# %matplotlib notebook
from matplotlib import pyplot as plt

from recommenders.utils.constants import SEED
from recommenders.utils.timer import Timer
from recommenders.datasets.download_utils import maybe_download, unzip_file
from recommenders.tuning.parameter_sweep import generate_param_grid
from recommenders.datasets.pandas_df_utils import LibffmConverter
from recommenders.utils.notebook_utils import store_metadata

print("System version: {}".format(sys.version))
print("Xlearn version: {}".format(xl.__version__))


def run_model():
    df_feature_original = pd.DataFrame(
        {
            "rating": [1, 0, 0, 1, 1],
            "field1": ["xxx1", "xxx2", "xxx4", "xxx4", "xxx4"],
            "field2": [3, 4, 5, 6, 7],
            "field3": [1.0, 2.0, 3.0, 4.0, 5.0],
            "field4": ["1", "2", "3", "4", "5"],
        }
    )

    converter = LibffmConverter().fit(df_feature_original, col_rating="rating")
    df_out = converter.transform(df_feature_original)
    # print(df_out)


    print(
        "There are in total {0} fields and {1} features.".format(
            converter.field_count, converter.feature_count
        )
    )

    # Model parameters
    LEARNING_RATE = 0.2
    LAMBDA = 0.002
    EPOCH = 10
    OPT_METHOD = "sgd"  # options are "sgd", "adagrad" and "ftrl"

    # The metrics for binary classification options are "acc", "prec", "f1" and "auc"
    # for regression, options are "rmse", "mae", "mape"
    METRIC = "auc"


    # Paths
    YAML_FILE_NAME = "xDeepFM.yaml"
    TRAIN_FILE_NAME = "cretio_tiny_train"
    VALID_FILE_NAME = "cretio_tiny_valid"
    TEST_FILE_NAME = "cretio_tiny_test"
    MODEL_FILE_NAME = "model.out"
    OUTPUT_FILE_NAME = "output.txt"

    tmpdir = TemporaryDirectory()
    print(tmpdir)

    data_path = tmpdir.name
    data_path = "./data/model_collaborative_filtering/m05"
    yaml_file = os.path.join(data_path, YAML_FILE_NAME)
    train_file = os.path.join(data_path, TRAIN_FILE_NAME)
    valid_file = os.path.join(data_path, VALID_FILE_NAME)
    test_file = os.path.join(data_path, TEST_FILE_NAME)
    model_file = os.path.join(data_path, MODEL_FILE_NAME)
    output_file = os.path.join(data_path, OUTPUT_FILE_NAME)

    assets_url = (
        "https://recodatasets.z20.web.core.windows.net/deeprec/xdeepfmresources.zip"
    )
    assets_file = maybe_download(assets_url, work_directory=data_path)
    unzip_file(assets_file, data_path)

    # Training task
    ffm_model = xl.create_ffm()  # Use field-aware factorization machine (ffm)
    ffm_model.setTrain(train_file)  # Set the path of training dataset
    ffm_model.setValidate(valid_file)  # Set the path of validation dataset

    # Parameters:
    #  0. task: binary classification
    #  1. learning rate: 0.2
    #  2. regular lambda: 0.002
    #  3. evaluation metric: auc
    #  4. number of epochs: 10
    #  5. optimization method: sgd
    param = {
        "task": "binary",
        "lr": LEARNING_RATE,
        "lambda": LAMBDA,
        "metric": METRIC,
        "epoch": EPOCH,
        "opt": OPT_METHOD,
    }

    # Start to train
    # The trained model will be stored in model.out
    with Timer() as time_train:
        ffm_model.fit(param, model_file)
    print(f"Training time: {time_train}")

    # Prediction task
    ffm_model.setTest(test_file)  # Set the path of test dataset
    ffm_model.setSigmoid()  # Convert output to 0-1

    # Start to predict
    # The output result will be stored in output.txt
    with Timer() as time_predict:
        ffm_model.predict(model_file, output_file)
    print(f"Prediction time: {time_predict}")

    with open(output_file) as f:
        predictions = f.readlines()

    with open(test_file) as f:
        truths = f.readlines()

    truths = np.array([float(truth.split(" ")[0]) for truth in truths])
    predictions = np.array([float(prediction.strip("")) for prediction in predictions])

    auc_score = roc_auc_score(truths, predictions)

    print(auc_score)

    store_metadata("auc_score", auc_score)

    param_dict = {"lr": [0.0001, 0.001, 0.01], "lambda": [0.001, 0.01, 0.1]}

    param_grid = generate_param_grid(param_dict)

    auc_scores = []

    with Timer() as time_tune:
        for param in param_grid:
            ffm_model = xl.create_ffm()
            ffm_model.setTrain(train_file)
            ffm_model.setValidate(valid_file)
            ffm_model.fit(param, model_file)

            ffm_model.setTest(test_file)
            ffm_model.setSigmoid()
            ffm_model.predict(model_file, output_file)

            with open(output_file) as f:
                predictions = f.readlines()

            with open(test_file) as f:
                truths = f.readlines()

            truths = np.array([float(truth.split(" ")[0]) for truth in truths])
            predictions = np.array(
                [float(prediction.strip("")) for prediction in predictions]
            )

            auc_scores.append(roc_auc_score(truths, predictions))

    print("Tuning by grid search takes {0:.2} min".format(time_tune.interval / 60))

    auc_scores = [float("%.4f" % x) for x in auc_scores]
    auc_scores_array = np.reshape(
        auc_scores, (len(param_dict["lr"]), len(param_dict["lambda"]))
    )

    auc_df = pd.DataFrame(
        data=auc_scores_array,
        index=pd.Index(param_dict["lr"], name="LR"),
        columns=pd.Index(param_dict["lambda"], name="Lambda"),
    )
    print(auc_df)

    fig, ax = plt.subplots()
    sns.heatmap(auc_df, cbar=False, annot=True, fmt=".4g")

    tmpdir.cleanup()


