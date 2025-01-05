from .recommender.model_collaborative_filtering import m01_deep_dive_spark_als as spark_als
from .recommender.model_collaborative_filtering import m02_deep_dive_baseline as base
from .recommender.model_collaborative_filtering import m03_deep_dive_cornac_bivae as cornac_bivae
from .recommender.model_collaborative_filtering import m04_deep_dive_cornac_bpr as cornac_bpr
from .recommender.model_collaborative_filtering import m05_deep_dive_fm as fm
from .recommender.model_collaborative_filtering import m06_deep_dive_lightfm as lightfm
from .recommender.model_collaborative_filtering import m07_deep_dive_lightgcn as lightgcn
from .recommender.model_collaborative_filtering import m08_deep_dive_multi_vae as multi_vae
from .recommender.model_collaborative_filtering import m09_deep_dive_ncf as ncf
from .recommender.model_collaborative_filtering import m10_deep_dive_rbm as rbm
from .recommender.model_collaborative_filtering import m11_deep_dive_sar as sar
from .recommender.model_collaborative_filtering import m12_deep_dive_standard_vae as standard_vae
from .recommender.model_collaborative_filtering import m13_deep_dive_svd_dive as svd_dive

#pd_sample = pd.read_pickle("./data/m03/movielens.pickle")
#pd_sample.to_csv("./data/m03/movielens_sample.csv")

# BiVAE > VAE-Multi >  Hybrid VAE > Standard VAE >  Max FM > NCF
# 2021  > 2019     >   2018        2018          > 2017
# 01
case_no = 1
print(f'---------run----------{__name__}' )

def run_test(case_no: int = 1):
    print("case no : ", case_no)
    if case_no == 1: # error # datapath
        spark_als.run_model()
    elif case_no == 2: # test ok # datapath
        base.run_model()
    elif case_no == 3: # test ok # datapath
        cornac_bivae.run_model()
    elif case_no == 4: # test ok # datapath fail (url setting)
        cornac_bpr.run_model()
    elif case_no == 5: # test ok # datapath
        fm.run_model()
    elif case_no == 6: # too take long
        lightfm.run_model()
    elif case_no == 7: # error
        lightgcn.run_model()
    elif case_no == 8: # test ok # model module
        # being Tested 2025.01.04
        #multi_vae.run_model_train()
        multi_vae.run_model_predict()
    elif case_no == 9: # test ok
        ncf.run_model()
    elif case_no == 10: # test ok
        rbm.run_model()
    elif case_no == 11: # test ok
       sar.run_model()
    elif case_no == 12: # test ok
        standard_vae.run_model()
    elif case_no == 13: # test ok
        svd_dive.run_model()


run_test(3)