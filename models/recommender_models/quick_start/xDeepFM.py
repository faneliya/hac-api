import os
import sys
from tempfile import TemporaryDirectory
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources, prepare_hparams
from recommenders.models.deeprec.models.xDeepFM import XDeepFMModel
from recommenders.models.deeprec.io.iterator import FFMTextIterator
from recommenders.utils.notebook_utils import store_metadata

print(f"System version: {sys.version}")
print(f"Tensorflow version: {tf.__version__}")

EPOCHS = 10
BATCH_SIZE = 4096
RANDOM_SEED = 42  # Set this to None for non-deterministic result

tmpdir = TemporaryDirectory()
data_path = tmpdir.name
yaml_file = os.path.join(data_path, r'xDeepFM.yaml')
output_file = os.path.join(data_path, r'output.txt')
train_file = os.path.join(data_path, r'cretio_tiny_train')
valid_file = os.path.join(data_path, r'cretio_tiny_valid')
test_file = os.path.join(data_path, r'cretio_tiny_test')

if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/deeprec/', data_path, 'xdeepfmresources.zip')

print('Demo with Criteo dataset')
hparams = prepare_hparams(yaml_file,
                          FEATURE_COUNT=2300000,
                          FIELD_COUNT=39,
                          cross_l2=0.01,
                          embed_l2=0.01,
                          layer_l2=0.01,
                          learning_rate=0.002,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          cross_layer_sizes=[20, 10],
                          init_value=0.1,
                          layer_sizes=[20,20],
                          use_Linear_part=True,
                          use_CIN_part=True,
                          use_DNN_part=True)
print(hparams)

model = XDeepFMModel(hparams, FFMTextIterator, seed=RANDOM_SEED)

model.fit(train_file, valid_file)

# ?
# model.save("./saved_models/xDeepFM.h5")

# check the predictive performance after the model is trained
result = model.run_eval(test_file)
print(result)

# Record results for tests - ignore this cell
store_metadata("auc", result["auc"])
store_metadata("logloss", result["logloss"])

# Cleanup
tmpdir.cleanup()