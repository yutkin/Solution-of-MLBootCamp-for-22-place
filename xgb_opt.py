import pprint
import xgboost
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from datetime import datetime
import teleloggingbot

dfx_train = pd.read_csv('./preds_train.csv', index_col=None)
dfy_train = pd.read_csv('./y_train.csv', index_col=None, header=None)

xgb_matrix = xgboost.DMatrix(dfx_train, dfy_train)
best_score = 1e6
best_param = None
counter = 0

def score(params):
  global xgb_matrix, best_score, best_param, counter
  
  num_round = int(params['n_estimators'])
  del params['n_estimators']
  cvresult = xgboost.cv(params, xgb_matrix, num_boost_round=num_round,
                    nfold=5, stratified=True, seed=42, verbose_eval=None,
                    early_stopping_rounds=50, metrics='logloss')
  score = cvresult['test-logloss-mean'].iloc[-1]
  params['n_estimators'] = len(cvresult)
  pprint.pprint(params)
  print("\t", score)
  
  if score < best_score:
    best_score = score
    best_param = params
    teleloggingbot.sendMsg('{}. New best logloss: {:.6}\n{}'.format(
      counter, best_score, pprint.pformat(best_param)))
  
  counter += 1
  if counter % 100 == 0:
    teleloggingbot.sendMsg('{}. Im working! Best logloss: {:.6}\n{}'.format(
      counter, best_score, pprint.pformat(best_param)))  
  return {'loss': score, 'status': STATUS_OK}

def optimize():
  space = {
    'n_estimators' : 5000,
    'learning_rate' : hp.choice('learning_rate', [0.008, 0.01, 0.015, 0.2]),
    'max_depth' : hp.choice('max_depth', np.arange(3, 15, dtype=int)),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 7, 1),
    'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma' : hp.quniform('gamma', 0.5, 10, 0.05),
    'reg_lambda' : hp.quniform('reg_lambda', 0.5, 10, 0.05),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'colsample_bylevel': hp.quniform('colsample_bylevel', 0.5, 1, 0.05),
    'max_delta_step': hp.quniform('max_delta_step', 0, 10, 0.05),
    'scale_pos_weight': hp.quniform('scale_pos_weight', 0, 1, 0.05),
    'objective': 'binary:logistic',
    'nthread' : 8,
    'silent' : 1
  }

  print('Start optimizing at ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
  try:
    fmin(score, space, algo=tpe.suggest, trials=Trials(), max_evals=500)
  except KeyboardInterrupt:
    pass
  res_str = '='*30
  res_str += '\nFinish optimizing at ' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
  res_str += '\nBest loss: ' + str(best_score)
  res_str += '\n'+pprint.pformat(best_param)
  teleloggingbot.sendMsg(res_str)
  print(res_str)

if __name__ == '__main__':
  optimize()