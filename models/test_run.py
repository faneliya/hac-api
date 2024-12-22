import pandas as pd

from .recommender_models.model_collaborative_filtering import m01_deep_dive_spark_als as spark_als
from .recommender_models.model_collaborative_filtering import m02_deep_dive_baseline as base
from .recommender_models.model_collaborative_filtering import m03_deep_dive_cornac_bivae as cornac_bivae
from .recommender_models.model_collaborative_filtering import m04_deep_dive_cornac_bpr as cornac_bpr
from .recommender_models.model_collaborative_filtering import m05_deep_dive_fm as fm
from .recommender_models.model_collaborative_filtering import m06_deep_dive_lightfm as lightfm
from .recommender_models.model_collaborative_filtering import m07_deep_dive_lightgcn as lightgcn
from .recommender_models.model_collaborative_filtering import m08_deep_dive_multi_vae as multi_vae
from .recommender_models.model_collaborative_filtering import m09_deep_dive_ncf as ncf
from .recommender_models.model_collaborative_filtering import m10_deep_dive_rbm as rbm
from .recommender_models.model_collaborative_filtering import m11_deep_dive_sar as sar

#pd_sample = pd.read_pickle("./data/m03/movielens.pickle")
#pd_sample.to_csv("./data/m03/movielens_sample.csv")

# 01
#spark_als.run_model()
# 02
#base.run_model()
# 03
#cornac_bivae.run_model()
# 04
#cornac_bivae.run_model()
# 05
#fm.run_model()
# 06
#lightfm.run_model()
# 07
lightgcn.run_model()

