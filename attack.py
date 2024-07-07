import argparse
from utils import *
from data.DataGeneration import *
from model.model import *

parser = argparse.ArgumentParser("VerticalFL")
parser.add_argument("-nparty", help="Number of clients", nargs='?', type=int, default=2)
parser.add_argument("-backbone", help="Backbone", nargs='?', type=str, default="resnet18")
args = parser.parse_args()


seedBasic()
strategy, AUTO = getStrategy()

_, preprocess_input = Classifiers.get(args.backbone)
(X_train, Y_train), (X_valid, Y_valid), (X_auxil, Y_auxil) = getCIFAR10(preprocess_input)

train_dataset = BaseDataGeneration(X_train, Y_train, args.batch, n_party=args.nparty)
valid_dataset = BaseDataGeneration(X_valid, Y_valid, args.batch, n_party=args.nparty)
auxil_dataset = BaseDataGeneration(X_auxil, Y_auxil, args.batch, n_party=args.nparty)

with strategy.scope():
    model, attackerClassifiers = model_factory(backbone = args.backbone,
                                               n_party = args.nparty)
    
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum),
                  loss = {'output': tf.keras.losses.CategoricalCrossentropy()},
                  metrics = {"output": [tf.keras.metrics.CategoricalAccuracy()]})

    for i in range(len(attackerClassifiers)):
        attackerClassifiers[i].compile(optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
                                       loss = {'output': tf.keras.losses.CategoricalCrossentropy()},
                                       metrics = {"output": [tf.keras.metrics.CategoricalAccuracy()]})

for i in range(len(attackerClassifiers)):
    attackerClassifiers[i].fit(auxil_dataset, validation_data = train_dataset, verbose = 1, epochs = 40)