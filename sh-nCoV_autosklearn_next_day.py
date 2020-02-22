# %%
import numpy as np
import pandas as pd
import gc
import time
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
# from joblib import dump, load
from numpy.core.defchararray import rstrip
pd.set_option('display.max_columns', None)
import autosklearn.regression
import sklearn.model_selection
import sklearn.metrics
import datetime
from sklearn.externals import joblib
# import mlflow #直接部署就不放代码了
# import mlflow.sklearn

# %%　#可以放最新的数据
train_df = pd.read_csv('data/shanghai-data-20200220.csv')
print(train_df.columns)
train_df.head()

# %%
X= train_df[:27].drop(['Date','sh_acc_p1d','sh_add'],axis=1)
y = train_df[:27].sh_add
X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=2)
automl = autosklearn.regression.AutoSklearnRegressor(seed=2)
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("MSE score", sklearn.metrics.mean_squared_error(y_test, y_hat))

print('train finished')

# %%
assembles=automl.get_models_with_weights()
params=automl.get_params()
model_final=automl.show_models()
statics=automl.sprint_statistics()
# %%
automl

# %% 保存模型
from sklearn.externals import joblib
#lr是一个LogisticRegression模型
joblib.dump(automl, 'autosklearn/model-add.model')

# %% 调入模型进行预测
# 0	3
# 8	3
# 5	3
# 2	2
# 0	2
# 0	2
# 1	2
# 0	2

automl1 = joblib.load('autosklearn/model-add.model')
data=pd.read_csv('data/shanghai-data-20200220.csv' )
# data=pd.read_csv('data/shanghai-data-%s.csv' % ((datetime.datetime.now().date()-datetime.timedelta(days=1)).strftime('%Y%m%d') ))

test_new= data[data.Date==20200213].drop(['Date','sh_acc_p1d','sh_add'],axis=1)#读进来适当做整数的
preds = automl1.predict(test_new)
print('predict finished')
print('Shanghai today 14th Feb 12-24hrs patients increased number is: ',preds)


test_new= data[data.Date==20200214].drop(['Date','sh_acc_p1d','sh_add'],axis=1)#读进来适当做整数的
preds = automl1.predict(test_new)
print('predict finished')
print('Shanghai today 15th Feb 12-24hrs patients increased number is: ',preds)


test_new= data[data.Date==20200215].drop(['Date','sh_acc_p1d','sh_add'],axis=1)#读进来适当做整数的
preds = automl1.predict(test_new)
print('predict finished')
print('Shanghai today 16th Feb 12-24hrs patients increased number is: ',preds)


test_new= data[data.Date==20200216].drop(['Date','sh_acc_p1d','sh_add'],axis=1)#读进来适当做整数的
preds = automl1.predict(test_new)
print('predict finished')
print('Shanghai today 17th Feb 12-24hrs patients increased number is: ',preds)

test_new= data[data.Date==20200217].drop(['Date','sh_acc_p1d','sh_add'],axis=1)#读进来适当做整数的
preds = automl1.predict(test_new)
print('predict finished')
print('Shanghai today 18th Feb 12-24hrs patients increased number is: ',preds)


test_new= data[data.Date==20200218].drop(['Date','sh_acc_p1d','sh_add'],axis=1)#读进来适当做整数的
preds = automl1.predict(test_new)
print('predict finished')
print('Shanghai today 19th Feb 12-24hrs patients increased number is: ',preds)


test_new= data[data.Date==20200219].drop(['Date','sh_acc_p1d','sh_add'],axis=1)#读进来适当做整数的
preds = automl1.predict(test_new)
print('predict finished')
print('Shanghai today 20th Feb 12-24hrs patients increased number is: ',preds)

test_new= data[data.Date==20200220].drop(['Date','sh_acc_p1d','sh_add'],axis=1)#读进来适当做整数的
preds = automl1.predict(test_new)
print('predict finished')
print('Shanghai today 21th Feb 12-24hrs patients increased number is: ',preds)


# %% 
# automl #缺省是1个小时
# AutoSklearnRegressor(delete_output_folder_after_terminate=True,
#                      delete_tmp_folder_after_terminate=True,
#                      disable_evaluator_output=False, ensemble_memory_limit=1024,
#                      ensemble_nbest=50, ensemble_size=50,
#                      exclude_estimators=None, exclude_preprocessors=None,
#                      get_smac_object_callback=None, include_estimators=None,
#                      include_preprocessors=None,
#                      initial_configurations_via_metalearning=25,
#                      logging_config=None, metadata_directory=None,
#                      ml_memory_limit=3072, n_jobs=None, output_folder=None,
#                      per_run_time_limit=360, resampling_strategy='holdout',
#                      resampling_strategy_arguments=None, seed=1,
#                      shared_mode=False, smac_scenario_args=None,
#                      time_left_for_this_task=3600, tmp_folder=None)
