# %%
import pandas as pd
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from numpy import concatenate
from math import sqrt
from keras.callbacks import EarlyStopping
from keras.models import load_model
# from keras import load_model
try:
  import cPickle as pickle
except ImportError:
  import pickle
# from numpy import log1p,expm1
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')
# %% 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):#1进1出
     # convert series to supervised learning
        n_vars = 1 if type(data) is list else data.shape[1]
        print(n_vars) #8
        df = pd.DataFrame(data)
        print(df.shape)
        # print('input df', df)
        cols, names = list(), list()
    	# input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):#TODO 这里将最后一行去除了，要预测的话要加上
            cols.append(df.shift(i))
            # print('cols', cols)
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # print('names',names)#['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)', 'var6(t-1)', 'var7(t-1)', 'var8(t-1)']
    	# forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i)) #df拼在右边，0-5第一行是NAN
            # print(cols)
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    	# put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        # print('processed agg',agg)
        return agg
# %%  训练和验证数据只到2月6日
def cs_to_sl():
    # load dataset
    dataset = pd.read_csv('data/sh-lstm-2-10.csv', header=0, index_col=0)
    values = dataset[:-4].values #最后一行需要预测，不含在预测中，-2，少2行参与建模，2/9， 2/8，2/7,2/6不含
    print(values.shape)
    # print(values) #显示全部的数据
    values = values.astype('float32')
    # print(values[:3]) #数据少，没必要
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1)) #TODO 使用指数
    # print('scaler',scaler, type(scaler))
    f = open('lstm_model/scaler', 'wb')
    pickle.dump(scaler, f)
    f.close()
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # print('reframe shape',reframed.shape)
    # print('reframe-',reframed)
    # drop columns we don't want to predict 一共6列，1 in 1 out，6*12，要预测的在6列，将7及之后的删除
    l=[]
    for i in range(values.shape[1]+1,values.shape[1]*2): #19-20~38： shape[1]+1,shape[1]*2 
        l.append(i)
    # l
    reframed.drop(reframed.columns[l], axis=1, inplace=True)
    # print('reframe output-',reframed)
    # print('-------')
    print(values[-1])
    # print(scaled[-1])
    return reframed,scaler,scaled[-1]
    # return reframed,scaled[-1]
# 原始数据：
# [[2.000e+00 2.000e+00 5.300e+00 8.310e+00 4.600e-01 2.240e+02]
#  [7.000e+00 5.000e+00 5.230e+00 1.074e+01 3.300e-01 2.580e+02]
#  [9.000e+00 7.000e+00 4.310e+00 1.184e+01 2.800e-01 3.200e+02]
#  [1.600e+01 1.200e+01 3.730e+00 1.110e+01 3.100e-01 4.950e+02]
#  [3.300e+01 2.600e+01 2.870e+00 3.890e+00 5.300e-01 5.720e+02]
#  [4.000e+01 3.200e+01 1.950e+00 1.130e+00 4.900e-01 6.180e+02]
#  [5.300e+01 4.100e+01 3.630e+00 6.600e-01 5.100e-01 6.960e+02]
#  [6.600e+01 5.000e+01 5.220e+00 4.300e-01 5.400e-01 8.920e+02]
#  [8.000e+01 6.100e+01 5.860e+00 3.200e-01 4.900e-01 1.590e+03]
#  [1.010e+02 7.600e+01 5.270e+00 2.600e-01 1.900e-01 1.905e+03]
#  [1.280e+02 8.400e+01 5.110e+00 2.400e-01 9.000e-02 2.639e+03]
#  [1.530e+02 1.010e+02 4.890e+00 2.400e-01 1.500e-01 3.215e+03]
#  [1.770e+02 1.090e+02 4.140e+00 2.400e-01 2.000e-01 4.109e+03]
#  [1.930e+02 1.130e+02 4.970e+00 4.600e-01 1.100e-01 5.142e+03]]
# scaled 0-1： 
# [[0.         0.         0.85677755 0.6956896  0.8222222  0.        ]
#  [0.02617801 0.02702703 0.8388747  0.90517235 0.5333333  0.00691338]
#  [0.03664921 0.04504504 0.6035805  0.99999994 0.4222222  0.01952013]
#  [0.07329843 0.0900901  0.45524293 0.9362069  0.48888886 0.0551037 ]
#  [0.16230366 0.21621622 0.23529404 0.31465515 0.97777766 0.07076047]
#  [0.19895287 0.2702703  0.         0.07672413 0.8888889  0.08011387]
#  [0.26701573 0.35135138 0.42966753 0.03620689 0.9333332  0.09597397]
#  [0.33507854 0.43243244 0.83631706 0.01637931 0.99999994 0.13582757]
#  [0.40837696 0.5315316  1.         0.00689655 0.8888889  0.27775517]
#  [0.5183246  0.6666667  0.84910476 0.00172414 0.2222222  0.34180558]
#  [0.65968585 0.7387388  0.80818415 0.         0.         0.49105325]
#  [0.7905759  0.8918919  0.7519181  0.         0.13333331 0.6081741 ]
#  [0.9162304  0.963964   0.5601022  0.         0.24444441 0.78995526]
#  [1.         1.         0.77237844 0.01896552 0.04444443 0.99999994]]
# 处理过后的数据：
# var1(t-1)  var2(t-1)  var3(t-1)  var4(t-1)  var5(t-1)  var6(t-1)   var1(t)
# 1    0.000000   0.000000   0.856778   0.695690   0.822222   0.000000  0.026178
# 2    0.026178   0.027027   0.838875   0.905172   0.533333   0.006913  0.036649
# 3    0.036649   0.045045   0.603580   1.000000   0.422222   0.019520  0.073298
# 4    0.073298   0.090090   0.455243   0.936207   0.488889   0.055104  0.162304
# 5    0.162304   0.216216   0.235294   0.314655   0.977778   0.070760  0.198953
# 6    0.198953   0.270270   0.000000   0.076724   0.888889   0.080114  0.267016
# 7    0.267016   0.351351   0.429668   0.036207   0.933333   0.095974  0.335079
# 8    0.335079   0.432432   0.836317   0.016379   1.000000   0.135828  0.408377
# 9    0.408377   0.531532   1.000000   0.006897   0.888889   0.277755  0.518325
# 10   0.518325   0.666667   0.849105   0.001724   0.222222   0.341806  0.659686
# 11   0.659686   0.738739   0.808184   0.000000   0.000000   0.491053  0.790576
# 12   0.790576   0.891892   0.751918   0.000000   0.133333   0.608174  0.916230
# 13   0.916230   0.963964   0.560102   0.000000   0.244444   0.789955  1.000000

def train_test(reframed):
    # split into train and test sets
    values = reframed.values
    n_train_hours = 13
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X,train_y,test_X,test_y

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto')

def fit_network(train_X,train_y,test_X,test_y,scaler):
    # print(train_X)
    model = Sequential() #这里就是50,1，非常简单两个层
    model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2])))
    # model.add(Dense(500))
    # model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
     # fit network
    history = model.fit(train_X, train_y, epochs=1000, batch_size=5, validation_data=(test_X, test_y), verbose=2, shuffle=False, callbacks=[earlystop])
    # history = model.fit(train_X, train_y, epochs=1000, batch_size=1, validation_data=(test_X, test_y), verbose=1, shuffle=False)
    # plot history
    model.save_weights('lstm_model/0211_weights') #保存模型
    model.save('lstm_model/0211_model') #
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    # make a prediction
    yhat = model.predict(test_X)
    # print(yhat)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))#3X6
    # invert scaling for forecast
    # print(test_X)
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    # inv_yhat = expm1(inv_yhat)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    # inv_y = expm1(test_X)
    inv_y = scaler.inverse_transform(test_X)
    inv_y = inv_y[:,0]
    # print(inv_y,inv_yhat)
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    return model
# %%
reframed,scaler,valid_lastest_for_predict = cs_to_sl()

train_X,train_y,test_X,test_y = train_test(reframed)

model=fit_network(train_X,train_y,test_X,test_y,scaler)

# %%
valid_lastest=valid_lastest_for_predict.reshape(1,1,19) #从（1,6）到(1,1,6)
last_day_predict = model.predict(valid_lastest)
# %%
valid_lastest_reshape=valid_lastest.reshape(valid_lastest.shape[0], valid_lastest.shape[2]) #从(1,1,6)到（1,6）
inv_yhat = concatenate((last_day_predict,valid_lastest_reshape[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)#反转为原始值
inv_yhat = inv_yhat[:,0]#只取预测值
print('Predicted next day nCov in Shanghai is: ',int(inv_yhat[0]))

# %% 重新预测
def cs_to_sl_p(last_line,scaler):
    # load dataset
    dataset = pd.read_csv('data/sh-lstm-2-10.csv', header=0, index_col=0)
    values = dataset[:last_line].values #最后一行需要预测，不含在预测中，-2，少2行参与建模，2/9， 2/8，2/7,2/6不含
    # print(values.shape)
    # print(values) #显示全部的数据
    # integer encode direction
    # encoder = LabelEncoder() #这里没有类型
    # values[:,4] = encoder.fit_transform(values[:,4]) #大数据可以这样吗？
    # ensure all data is float
    values = values.astype('float32')
    # print(values[:3]) #数据少，没必要
    # normalize features
    # scaler = MinMaxScaler(feature_range=(0, 1)) #TODO 使用指数
    scaled = scaler.fit_transform(values)
    # scaled=log1p(values)
    # print('scaled-',scaled)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # print('reframe shape',reframed.shape)
    # print('reframe-',reframed)
    # drop columns we don't want to predict 一共6列，1 in 1 out，6*12，要预测的在6列，将7及之后的删除
    l=[]
    for i in range(values.shape[1]+1,values.shape[1]*2): #19-20~38： shape[1]+1,shape[1]*2 
        l.append(i)
    # l
    reframed.drop(reframed.columns[l], axis=1, inplace=True)
    # print('reframe output-',reframed)
    print('-------')
    print(values[-1])
    # print(scaled[-1])
    return reframed,scaler,scaled[-1]
# %% #预测
_,_,valid_lastest_for_predict = cs_to_sl_p(-3,scaler)
valid_lastest=valid_lastest_for_predict.reshape(1,1,19) #从（1,6）到(1,1,6)
last_day_predict = model.predict(valid_lastest)
valid_lastest_reshape=valid_lastest.reshape(valid_lastest.shape[0], valid_lastest.shape[2]) #从(1,1,6)到（1,6）
inv_yhat = concatenate((last_day_predict,valid_lastest_reshape[:, 1:]), axis=1)
# inv_yhat = expm1(inv_yhat)#反转为原始值
inv_yhat = scaler.inverse_transform(inv_yhat)#反转为原始值
inv_yhat = inv_yhat[:,0]#只取预测值
print('Predicted next day nCov in Shanghai is: ',int(inv_yhat[0]))
# 292-298 预测
#  %% 调用保存模型
#恢复scaler
f = open('lstm_model/scaler', 'rb')
scaler1 = pickle.load(f)
f.close()
#恢复模型
model1 = Sequential() 
model1.add(LSTM(500, input_shape=(1, 19)))#输入是19
model1.add(Dense(1))
model1.load_weights('lstm_model/0211_weights', by_name=False)

_,_,valid_lastest_for_predict = cs_to_sl_p(-5,scaler1)
# _,_,valid_lastest_for_predict = cs_to_sl_p(-3,scaler)
valid_lastest=valid_lastest_for_predict.reshape(1,1,19) #从（1,6）到(1,1,6)
last_day_predict = model1.predict(valid_lastest)
valid_lastest_reshape=valid_lastest.reshape(valid_lastest.shape[0], valid_lastest.shape[2]) #从(1,1,6)到（1,6）
inv_yhat = concatenate((last_day_predict,valid_lastest_reshape[:, 1:]), axis=1)
# inv_yhat = expm1(inv_yhat)#反转为原始值
inv_yhat = scaler.inverse_transform(inv_yhat)#反转为原始值
inv_yhat = inv_yhat[:,0]#只取预测值
print('saved model：Predicted next day nCov in Shanghai is: ',int(inv_yhat[0]))

_,_,valid_lastest_for_predict = cs_to_sl_p(-4,scaler1)
# _,_,valid_lastest_for_predict = cs_to_sl_p(-3,scaler)
valid_lastest=valid_lastest_for_predict.reshape(1,1,19) #从（1,6）到(1,1,6)
last_day_predict = model1.predict(valid_lastest)
valid_lastest_reshape=valid_lastest.reshape(valid_lastest.shape[0], valid_lastest.shape[2]) #从(1,1,6)到（1,6）
inv_yhat = concatenate((last_day_predict,valid_lastest_reshape[:, 1:]), axis=1)
# inv_yhat = expm1(inv_yhat)#反转为原始值
inv_yhat = scaler.inverse_transform(inv_yhat)#反转为原始值
inv_yhat = inv_yhat[:,0]#只取预测值
print('saved model：Predicted next day nCov in Shanghai is: ',int(inv_yhat[0]))

_,_,valid_lastest_for_predict = cs_to_sl_p(-3,scaler1)
# _,_,valid_lastest_for_predict = cs_to_sl_p(-3,scaler)
valid_lastest=valid_lastest_for_predict.reshape(1,1,19) #从（1,6）到(1,1,6)
last_day_predict = model1.predict(valid_lastest)
valid_lastest_reshape=valid_lastest.reshape(valid_lastest.shape[0], valid_lastest.shape[2]) #从(1,1,6)到（1,6）
inv_yhat = concatenate((last_day_predict,valid_lastest_reshape[:, 1:]), axis=1)
# inv_yhat = expm1(inv_yhat)#反转为原始值
inv_yhat = scaler.inverse_transform(inv_yhat)#反转为原始值
inv_yhat = inv_yhat[:,0]#只取预测值
print('saved model：Predicted next day nCov in Shanghai is: ',int(inv_yhat[0]))
# 292-298 预测

_,_,valid_lastest_for_predict = cs_to_sl_p(-2,scaler)
valid_lastest=valid_lastest_for_predict.reshape(1,1,19) #从（1,6）到(1,1,6)
last_day_predict = model1.predict(valid_lastest)
valid_lastest_reshape=valid_lastest.reshape(valid_lastest.shape[0], valid_lastest.shape[2]) #从(1,1,6)到（1,6）
inv_yhat = concatenate((last_day_predict,valid_lastest_reshape[:, 1:]), axis=1)
# inv_yhat = expm1(inv_yhat)#反转为原始值
inv_yhat = scaler.inverse_transform(inv_yhat)#反转为原始值
inv_yhat = inv_yhat[:,0]#只取预测值
print('Predicted next day nCov in Shanghai is: ',int(inv_yhat[0]))
# 295-302，实际302#模型已备份

_,_,valid_lastest_for_predict = cs_to_sl_p(-1,scaler)
valid_lastest=valid_lastest_for_predict.reshape(1,1,19) #从（1,6）到(1,1,6)
last_day_predict = model1.predict(valid_lastest)
valid_lastest_reshape=valid_lastest.reshape(valid_lastest.shape[0], valid_lastest.shape[2]) #从(1,1,6)到（1,6）
inv_yhat = concatenate((last_day_predict,valid_lastest_reshape[:, 1:]), axis=1)
# inv_yhat = expm1(inv_yhat)#反转为原始值
inv_yhat = scaler.inverse_transform(inv_yhat)#反转为原始值
inv_yhat = inv_yhat[:,0]#只取预测值
print('Predicted next day nCov in Shanghai is: ',int(inv_yhat[0]))


# %% 这个方法也可以。
# model2=load_model('lstm_model/0211_model')
# last_day_predict = model2.predict(valid_lastest)
# valid_lastest_reshape=valid_lastest.reshape(valid_lastest.shape[0], valid_lastest.shape[2]) #从(1,1,6)到（1,6）
# inv_yhat = concatenate((last_day_predict,valid_lastest_reshape[:, 1:]), axis=1)
# # inv_yhat = expm1(inv_yhat)#反转为原始值
# inv_yhat = scaler.inverse_transform(inv_yhat)#反转为原始值
# inv_yhat = inv_yhat[:,0]#只取预测值
# print('saved model：Predicted next day nCov in Shanghai is: ',int(inv_yhat[0]))
