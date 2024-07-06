import argparse
from utils import *
from data.DataGeneration import *
from model.model import *

parser = argparse.ArgumentParser("VerticalFL")
parser.add_argument("-epochs", help="Number of local epochs", nargs='?', type=int, default=10)
parser.add_argument("-batch", help="Batch size", nargs='?', type=int, default=128)
parser.add_argument("-lr", help="Learning rate", nargs='?', type=float, default=1e-2)
parser.add_argument("-momentum", help="Batch size", nargs='?', type=float, default=0.9)
args = parser.parse_args()

seedBasic()
strategy, AUTO = getStrategy()

(X_train, Y_train), (X_valid, Y_valid), (X_auxil, Y_auxil) = getCIFAR10()

train_dataset = BaseDataGeneration(X_train, Y_train, args.batch)
valid_dataset = BaseDataGeneration(X_valid, Y_valid, args.batch)
auxil_dataset = BaseDataGeneration(X_auxil, Y_auxil, args.batch)

with strategy.scope():
    model, attackerClassifiers = model_factory()
    
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum),
                  loss = {'output': tf.keras.losses.CategoricalCrossentropy()},
                  metrics = {"output": [tf.keras.metrics.CategoricalAccuracy()]})
    
H = model.fit(train_dataset,
              validation_data = valid_dataset,
              verbose = 1,
              epochs = args.epochs)