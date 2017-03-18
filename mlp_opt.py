import pprint
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

from hyperopt import Trials, STATUS_OK, tpe, hp, fmin

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from datetime import datetime
import teleloggingbot

dfx_train = pd.read_csv('./x_train_my.csv', index_col=None)
dfy_train = pd.read_csv('./y_train.csv', header=None)

best_score = 1e6
best_param = None
n_iter = 0


def score(param):
    global best_score, best_param, n_iter, Xtr, Xts, Ytr, Yts

    kf = StratifiedKFold(n_splits=4, random_state=42, shuffle=True)
    early_stops = []
    losses = []
    xtrain = dfx_train.values
    ytain = dfy_train.values
    for train_index, test_index in kf.split(xtrain, ytrain.ravel()):
        X_train, X_test = xtrain[train_index], xtrain[test_index]
        y_train, y_test = ytrain[train_index], ytrain[test_index]
        
        model = Sequential()
        model.add(Dense(param['1layer_neurons'], init='he_uniform',
            input_dim=X_train.shape[1]))
        model.add(Activation('relu'))
        model.add(Dropout(param['1layer_dropout']))

        if param['2_layer']:
            model.add(Dense(param['2_layer']['2layer_neurons']))
            model.add(Dropout(param['2_layer']['2layer_dropout']))
            model.add(Activation('relu'))
        model.add(Dense(1, activation='sigmoid', init='he_uniform'))

        model.compile(
            loss='binary_crossentropy',
            optimizer=param['optimizer'],
            metrics=[])

        early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=5,
            verbose=0,
            mode='auto')

        model.fit(X_train, y_train,
                  batch_size=param['batch_size'],
                  nb_epoch=500,
                  verbose=0,
                  validation_split=0,
                  validation_data=(X_test, y_test),
                  callbacks=[early_stop])

        oof_loss = model.evaluate(X_test, y_test,
            batch_size=param['batch_size'], verbose=0)
        losses.append(oof_loss)
        early_stops.append(early_stop.stopped_epoch)
    
    loss = np.mean(losses)
    param['epoch'] = int(np.mean(early_stops))

    n_iter += 1
    print('\t', n_iter, 'Logloss:', loss)
    pprint.pprint(param)
    print('=' * 30)
    if loss < best_score:
        best_score = loss
        best_param = param
        teleloggingbot.sendMsg('{}. New best logloss: {:.6}\n{}'.format(
            n_iter, loss, pprint.pformat(best_param)))

    return {'loss': loss, 'status': STATUS_OK}


def optimize():
    global best_score, best_param

    space = {
        '1layer_neurons': hp.choice('1layer_neurons',
            [32, 64, 128, 256]),
        '1layer_dropout': hp.choice('1layer_dropout',
            [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]),
        '2_layer': hp.choice('2_layer', [
            ({
                '2layer_neurons': hp.choice('2layer_neurons',
                    [32, 64, 128, 256]),
                '2layer_dropout': hp.choice('2layer_dropout',
                    [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]),
            }),
            (None)
        ]),
        'optimizer': hp.choice('optimizer', ['rmsprop', 'adam']),
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256, 512]),
    }

    print('Start optimizing at ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    try:
        fmin(score, space, algo=tpe.suggest, trials=Trials(), max_evals=600)
    except KeyboardInterrupt:
        pass
    res_str = '\n'
    res_str += '=' * 30
    res_str += '\nFinish optimizing at ' + \
        str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    res_str += '\nBest loss: ' + str(best_score)
    res_str += '\n' + pprint.pformat(best_param,)
    res_str = '='*30
    teleloggingbot.sendMsg(res_str)

    print(res_str)

if __name__ == '__main__':
    optimize()
