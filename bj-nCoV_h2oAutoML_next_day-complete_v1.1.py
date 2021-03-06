# To add a new cell, type '# %%' http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import gc
# import time
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.filterwarnings('ignore')
# from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, f1_score, confusion_matrix, accuracy_score
# from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)
import h2o
from h2o.automl import H2OAutoML
h2o.init()
# import mlflow
# %%　１４个变量
train_df = pd.read_csv('data/bj-nCoV-2-10-pre-add.csv',nrows=17)
print(train_df.columns)
train_df.head()
test_x=train_df[13:].drop(['date','totalPlus1d'],axis=1)
y_test=train_df[13:].totalPlus1d
#这里是使用全部进行训练，导致最后一个效果很好
train=train_df[:13].drop(['date'],axis=1)
train= h2o.H2OFrame.from_python(train)
y = 'totalPlus1d'
#增加训练时间才可以有好的模型出来。
aml = H2OAutoML(max_runtime_secs=600,seed=1,verbosity='info',include_algos = ["DeepLearning"])#max_runtime_secs=4800
aml.train(y=y, training_frame=train)
# Look at Leaderboard (will have more or fewer models, depending on the hardware used)
print(aml.leaderboard)
print('train finished')
# %%
preds0 = aml.leader.predict(h2o.H2OFrame.from_python(test_x))
print('predict finished')
from math import sqrt
from sklearn.metrics import mean_squared_error
preds = preds0.as_data_frame(use_pandas=True, header=True) 
print(y_test, preds2)
rmse = sqrt(mean_squared_error(y_test, preds))
print('Test RMSE: %.3f' % rmse)

# %%
saved_model=h2o.save_model(aml.leader, 
          path="./bj-ncov_model_pre", force=True)# train_df都是含Id的
print(saved_model)

aml.leader
data=pd.read_csv('data/bj-nCoV-2-10-pre-add.csv')
test_new= h2o.H2OFrame.from_python(data.tail(1).drop(['date','totalPlus1d'],axis=1))
preds = aml.leader.predict(test_new)
print('predict finished')
pred = preds.as_data_frame(use_pandas=True, header=True) 
pred0=pred.predict
print('beijing today 5th Feb 12-24hrs patients number is: ',int(pred0) )

# %% 调入模型进行预测-先校核
saved_model0 = h2o.load_model(saved_model)
# saved_model0 = h2o.load_model('bj-ncov_model_pre/DeepLearning_grid_1_AutoML_20200208_153836_model_38')
data=pd.read_csv('data/bj-nCoV-2-10-pre-add.csv')
test_new= h2o.H2OFrame.from_python(data[data.date=='2月4日'].drop(['date','totalPlus1d'],axis=1))

preds = saved_model0.predict(test_new)
print('predict finished')
pred = preds.as_data_frame(use_pandas=True, header=True) 
pred0=pred.predict
print('Beijing today 5th Feb 12-24hrs patients number is: ',round(pred0) )

test_new= h2o.H2OFrame.from_python(data[data.date=='2月5日'].drop(['date','totalPlus1d'],axis=1))
preds = saved_model0.predict(test_new)
print('predict finished')
pred = preds.as_data_frame(use_pandas=True, header=True) 
pred0=pred.predict
print('Beijing today 6th Feb 12-24hrs patients number is: ',round(pred0) )

test_new= h2o.H2OFrame.from_python(data[data.date=='2月6日'].drop(['date','totalPlus1d'],axis=1))
preds = saved_model0.predict(test_new)
print('predict finished')
pred = preds.as_data_frame(use_pandas=True, header=True) 
pred0=pred.predict
print('Beijing today 7th Feb 12-24hrs patients number is: ',round(pred0) )

test_new= h2o.H2OFrame.from_python(data[data.date=='2月7日'].drop(['date','totalPlus1d'],axis=1))
preds = saved_model0.predict(test_new)
print('predict finished')
pred = preds.as_data_frame(use_pandas=True, header=True) 
pred0=pred.predict
print('Beijing today 8th Feb 12-24hrs patients number is: ',round(pred0) )

test_new= h2o.H2OFrame.from_python(data[data.date=='2月8日'].drop(['date','totalPlus1d'],axis=1))
preds = saved_model0.predict(test_new)
print('predict finished')
pred = preds.as_data_frame(use_pandas=True, header=True) 
pred0=pred.predict
print('Beijing today 9th Feb 12-24hrs patients number is: ',round(pred0) )

test_new= h2o.H2OFrame.from_python(data[data.date=='2月9日'].drop(['date','totalPlus1d'],axis=1))
preds = saved_model0.predict(test_new)
print('predict finished')
pred = preds.as_data_frame(use_pandas=True, header=True) 
pred0=pred.predict
print('Beijing today 10th Feb 12-24hrs patients number is: ',round(pred0) )

test_new= h2o.H2OFrame.from_python(data[data.date=='2月10日'].drop(['date','totalPlus1d'],axis=1))
preds = saved_model0.predict(test_new)
print('predict finished')
pred = preds.as_data_frame(use_pandas=True, header=True) 
pred0=pred.predict
print('Beijing today 11th Feb 12-24hrs patients number is: ',round(pred0) )


# %%
