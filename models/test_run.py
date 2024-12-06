import pandas as pd

from .recommender_models.model_collaborative_filtering import m02_deep_dive_baseline as base
from .recommender_models.model_collaborative_filtering import m03_deep_dive_cornac_bivae as cornac_bivae
from .recommender_models.model_collaborative_filtering import m05_deep_dive_fm as fm
from .recommender_models.model_collaborative_filtering import m06_deep_dive_lightfm as lightfm

#pd_sample = pd.read_pickle("./data/m03/movielens.pickle")
#pd_sample.to_csv("./data/m03/movielens_sample.csv")

#base.run_model()
#cornac_bivae.run_model()
#fm.run_model()
lightfm.run_model()


