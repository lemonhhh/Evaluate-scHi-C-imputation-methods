import torch
print(torch.cuda.is_available())
import sys
#使用shareb/mliu/evaluate_impute/tools/Higashi/Higashi这个路径下的higashi包
sys.path.append('/shareb/mliu/evaluate_impute/tools/Higashi/Higashi')
from higashi.Higashi_wrapper import *



config = "/shareb/mliu/evaluate_impute/tools/Higashi/data/config.JSON"
higashi_model = Higashi(config)




higashi_model.process_data()
higashi_model.prep_model()
higashi_model.train_for_embeddings()
higashi_model.train_for_imputation_nbr_0()
higashi_model.impute_no_nbr()

# higashi_model.train_for_imputation_with_nbr()
# higashi_model.impute_with_nbr()