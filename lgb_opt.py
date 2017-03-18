import pprint
import lightgbm as lgb
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from datetime import datetime
import teleloggingbot

dfx_train = pd.read_csv('./preds_train.csv', index_col=None)
dfy_train = pd.read_csv('./y_train.csv', index_col=None, header=None)

lgb_matrix = lgb.Dataset(dfx_train.as_matrix(), dfy_train.as_matrix().ravel())

best_score = 1e6
best_param = None
counter = 0

def score(params):
  global lgb_matrix, best_score, best_param, counter
  cvresult = lgb.cv(params, lgb_matrix,
    num_boost_round=params['num_iterations'],
    nfold=10,
    stratified=True,
    early_stopping_rounds=50,
    show_stdv=False,
    seed=42,
    verbose_eval=None)
  params['num_iterations'] = len(cvresult['binary_logloss-mean'])
  pprint.pprint(params)
  msg = 'Done lightgbm CV! logloss: {} +- {}\n{}'.format(
    cvresult['binary_logloss-mean'][-1],
    round(cvresult['binary_logloss-stdv'][-1], 8),
    pprint.pformat(params))

  score = cvresult['binary_logloss-mean'][-1]
  print("\t", score)
  
  counter += 1
  if counter % 100 == 0:
    teleloggingbot.sendMsg('{}. Im working! Best logloss: {:.6}\n{}'.format(
      counter, best_score, pprint.pformat(best_param)))  
  
  if score < best_score:
    best_score = score
    best_param = params
    teleloggingbot.sendMsg('{}. New best logloss: {:.6}\n{}'.format(
      counter, best_score, pprint.pformat(best_param)))

  return {'loss': score, 'status': STATUS_OK}

def optimize(trials):
  space = {
    'bagging_fraction': hp.quniform('bagging_fraction', 0.1, 1, 0.05),
    'bagging_freq': hp.choice('bagging_freq', np.arange(0, 5, dtype=int)),
    'feature_fraction': hp.quniform('feature_fraction', 0.5, 1, 0.05),
    'learning_rate': hp.quniform('learning_rate', 0.001, 0.3, 0.002),
    'max_depth': hp.choice('max_depth', [-1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    'metric': ['binary_logloss'],
    'num_leaves': hp.choice('num_leaves', [15, 31, 65, 127, 255, 511, 1023]),
    'max_bin': hp.choice('max_bin', [15, 31, 65, 127, 255, 511, 1023]),
    'boosting': hp.choice('boosting', ['gbdt', 'dart']),
    'lambda_l2': hp.quniform('lambda_l2', 0, 10, 0.05),
    'num_iterations': 5000,
    'num_threads': 4,
    'objective': 'binary',
    'verbose': 0
  }

  print('Start optimizing at ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
  try:
    fmin(score, space, algo=tpe.suggest, trials=Trials(), max_evals=400)
  except KeyboardInterrupt:
    pass

  res_str = '='*30
  res_str += '\nFinish optimizing at ' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
  res_str += '\nBest loss: ' + str(best_score)
  res_str += '\n'+pprint.pformat(best_param)
  teleloggingbot.sendMsg(res_str)
  print(res_str)

trials = Trials()

optimize(trials)