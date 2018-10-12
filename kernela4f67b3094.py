# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test= pd.read_csv('../input/test.csv')
need = [ 'assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 
                'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks', 
                'longestKill', 'maxPlace', 'numGroups', 'rideDistance', 
                'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 
                'weaponsAcquired', 'winPoints','winPlacePerc'] 
try_data=train[need]
from sklearn.model_selection import train_test_split
x_data=try_data.drop('winPlacePerc',axis=1)
y_data=try_data['winPlacePerc']
X_train,X_test,y_train,y_test= train_test_split(x_data,y_data,test_size=0.3,random_state=0)
import lightgbm as lgb
dtrain = lgb.Dataset(X_train.values,y_train.values) 
dval   = lgb.Dataset(X_test.values,y_test.values,reference=dtrain) 

params = {
        'task':'train', 
        'num_leaves': 255, 
        'objective': 'regression',
        'metric': 'mape',
        'min_data_in_leaf': 30,
        'learning_rate': 0.05,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5, 
        'max_bin':128,
        'num_threads': 64,
        'random_state':100
    }  
lgb_model_step  = lgb.train(params, dtrain, num_boost_round=2000,valid_sets=[dtrain,dval], early_stopping_rounds=30)
need_test=['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 
                'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks', 
                'longestKill', 'maxPlace', 'numGroups', 'rideDistance', 
                'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 
                'weaponsAcquired', 'winPoints']
import pandas as pd 
test= pd.read_csv('../input/test.csv')
try_test=test[need_test]
result=lgb_model_step.predict(try_test,num_iteration=lgb_model_step.best_iteration)
pd.DataFrame({'ID':test['Id'],'winPlacePerc':result}).to_csv('submission2.csv', index=False)
# Any results you write to the current directory are saved as output.