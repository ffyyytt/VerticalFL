import os
import random
import numpy as np
import tensorflow as tf

from keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder

def seedBasic(seed=1312):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def getStrategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    except ValueError:
        tpu = None
        gpus = tf.config.experimental.list_logical_devices("GPU")

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        tf.config.set_soft_device_placement(True)

        print('Running on TPU ', tpu.master())
    elif len(gpus) > 0:
        strategy = tf.distribute.MirroredStrategy(gpus)
        print('Running on ', len(gpus), ' GPU(s) ')
    else:
        strategy = tf.distribute.get_strategy()
        print('Running on CPU')

    print("Number of accelerators: ", strategy.num_replicas_in_sync)

    AUTO = tf.data.experimental.AUTOTUNE
    return strategy, AUTO

def getCIFAR10():
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()
    auxilID = sum([random.choices(np.where(Y_train==i)[0].tolist(), k = 40) for i in range(10)], [])
    X_auxil = X_train[auxilID]
    Y_auxil = Y_train[auxilID]


    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    Y_auxil = Y_auxil.astype('float32')


    encoder = OneHotEncoder()
    encoder.fit(Y_train)
    Y_train = encoder.transform(Y_train).toarray()
    Y_valid = encoder.transform(Y_valid).toarray()
    Y_auxil = encoder.transform(Y_auxil).toarray()

    return (X_train, Y_train), (X_valid, Y_valid), (X_auxil, Y_auxil)