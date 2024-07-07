import os
import random
import numpy as np
import tensorflow as tf

from tqdm import *
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

def getCIFAR10(preprocess_input):
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()
    auxilID = sum([random.choices(np.where(Y_train==i)[0].tolist(), k = 400) for i in range(10)], [])
    X_auxil = X_train[auxilID]
    Y_auxil = Y_train[auxilID]


    X_train = preprocess_input(X_train.astype('float32'))
    X_valid = preprocess_input(X_valid.astype('float32'))
    Y_auxil = preprocess_input(Y_auxil.astype('float32'))


    encoder = OneHotEncoder()
    encoder.fit(Y_train)
    Y_train = encoder.transform(Y_train).toarray()
    Y_valid = encoder.transform(Y_valid).toarray()
    Y_auxil = encoder.transform(Y_auxil).toarray()

    return (X_train, Y_train), (X_valid, Y_valid), (X_auxil, Y_auxil)

def compute_saliency_map(dataset, model):
    saliency_maps = []
    for data in tqdm(dataset):
        input_image= data[0]
        for k, v in input_image.items():
            input_image[k] = tf.convert_to_tensor(v)
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            predictions = model(input_image)
            top_class = tf.argmax(predictions[0])
            top_class_score = predictions[:, top_class]
        
        gradients = tape.gradient(top_class_score, input_image)
        saliency_maps.append(tf.reduce_max(tf.abs(gradients["image_0"]), axis=-1).numpy())
    saliency_maps = np.concatenate(gradients, axis=0)
    return saliency_maps

def getMaxWindow(data, window_size):
    windows = []
    for i in range(data.shape[0] - window_size + 1):
        for j in range(data.shape[1] - window_size + 1):
            windows.append([[i, j, window_size], np.mean(data[i:i + window_size, j:j + window_size])])
    return max(windows, key=lambda x: x[1])