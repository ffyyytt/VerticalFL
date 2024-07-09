import argparse
from utils import *
from data.DataGeneration import *
from model.model import *
from collections import Counter

parser = argparse.ArgumentParser("VerticalFL")
parser.add_argument("-epochs", help="Number of local epochs", nargs='?', type=int, default=100)
parser.add_argument("-batch", help="Batch size", nargs='?', type=int, default=128)
parser.add_argument("-lr", help="Learning rate", nargs='?', type=float, default=1e-2)
parser.add_argument("-momentum", help="Batch size", nargs='?', type=float, default=0.9)
parser.add_argument("-backbone", help="Backbone", nargs='?', type=str, default="resnet18")

parser.add_argument("-windowSize", help="Trigger size", nargs='?', type=int, default=5)
parser.add_argument("-nparty", help="Number of clients", nargs='?', type=int, default=2)
parser.add_argument("-p", help="Percentage subset of the source class", nargs='?', type=float, default=0.5)

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
        attackerClassifiers[i].compile(optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum),
                                       loss = {'output': tf.keras.losses.CategoricalCrossentropy()},
                                       metrics = {"output": [tf.keras.metrics.CategoricalAccuracy()]})

targetClass = 1
sourceClass = 0

positions = {}
for i in range(len(attackerClassifiers)):
    attackerClassifiers[i].fit(auxil_dataset, validation_data = train_dataset, verbose = 1, epochs = args.epochs)
    saliency_maps = compute_saliency_map(train_dataset, attackerClassifiers[i])
    positions[i] = np.array([getMaxWindow(saliency_maps[i], args.windowSize)[0] for i in trange(len(saliency_maps))])

attackDataGeneration = AttackDataGeneration(model, args.p, X_train, Y_train, positions, targetClass, sourceClass, args.windowSize, 
                                            args.batch, strategy, args.lr, args.momentum, args.epochs, n_party=args.nparty)

H = model.fit(attackDataGeneration,
              validation_data = valid_dataset,
              verbose = 1,
              epochs = args.epochs)

yPred = model.predict(valid_dataset)
yPred = np.argmax(yPred, axis=1)
YValid = np.argmax(Y_valid, axis=1)
print("MTA:", np.mean(yPred == YValid))
print("ASR:", np.mean(yPred[np.where(YValid==sourceClass)] == targetClass))
print("ASR:", Counter(yPred[np.where(YValid==sourceClass)]))
print("ASR:", Counter(yPred[np.where(YValid==targetClass)]))
print("ASR:", Counter(yPred))