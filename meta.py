#!/usr/bin/env python3
#
# This is the script I use to tune the hyper-parameters automatically.
# 进行参数空间搜索的函数
#
import subprocess

import hyperopt
import logging

min_y = 0
min_c = None

logging.basicConfig(level=logging.DEBUG,  #控制台打印的日志级别
                    filename='result/20211012_02.log',
                    filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

def trial(hyperpm):
    global min_y, min_c
    # Plz set nbsz manually. Maybe a larger value if you have a large memory.
    cmd = 'python main.py --dataname cora --nbsz 30'
    cmd = 'CUDA_VISIBLE_DEVICES=5 ' + cmd
    for k in hyperpm:
        v = hyperpm[k]
        cmd += ' --' + k
        if int(v) == v:
            cmd += ' %d' % int(v)
        else:
            cmd += ' %g' % float('%.1e' % float(v))
    try:
        val, tst = eval(subprocess.check_output(cmd, shell=True))
    except subprocess.CalledProcessError:
        print('...')
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL}
    logging.info('val=%5.2f%% tst=%5.2f%% @ %s' % (val * 100, tst * 100, cmd))
    print('val=%5.2f%% tst=%5.2f%% @ %s' % (val * 100, tst * 100, cmd))
    score = -val
    if score < min_y:
        min_y, min_c = score, cmd
    return {'loss': score, 'status': hyperopt.STATUS_OK}


space = {'lr': hyperopt.hp.loguniform('lr', -8, 0),
         'reg': hyperopt.hp.loguniform('reg', -10, 0),
         'nlayer': 4,
         'ncaps': 7,
         'nhidden': hyperopt.hp.quniform('nhidden', 2, 32, 2),
         'dropout': hyperopt.hp.uniform('dropout', 0, 1),
         'routit': 6}
hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=1000)
logging.info('>>>>>>>>>> val=%5.2f%% @ %s' % (-min_y * 100, min_c))
print('>>>>>>>>>> val=%5.2f%% @ %s' % (-min_y * 100, min_c))