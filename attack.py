import argparse
from utils import *
from model.model import *
from data.DataGeneration import *
from callbacks.DetectCallback import *
from collections import Counter

parser = argparse.ArgumentParser("VerticalFL")
parser.add_argument("-epochs", help="Number of local epochs", nargs='?', type=int, default=100)
parser.add_argument("-batch", help="Batch size", nargs='?', type=int, default=32)
parser.add_argument("-lr", help="Learning rate", nargs='?', type=float, default=1e-2)
parser.add_argument("-momentum", help="Batch size", nargs='?', type=float, default=0.9)
parser.add_argument("-backbone", help="Backbone", nargs='?', type=str, default="resnet18")
parser.add_argument("-n_attackers", help="Number of attackers", nargs='?', type=int, default=1)
parser.add_argument("-windowSize", help="Trigger size", nargs='?', type=int, default=5)
parser.add_argument("-nparty", help="Number of clients", nargs='?', type=int, default=2)
parser.add_argument("-p", help="Percentage subset of the source class", nargs='?', type=float, default=0.5)
parser.add_argument("-selection", help="Optimal Selection", nargs='?', type=bool, default=True)

args = parser.parse_args()


seedBasic()
strategy, AUTO = getStrategy()

_, preprocess_input = Classifiers.get(args.backbone)
(X_train, Y_train), (X_valid, Y_valid), (X_auxil, Y_auxil) = getCIFAR10(preprocess_input)

X_train = X_train[:1000]
Y_train = Y_train[:1000]

X_valid = X_valid[:1000]
Y_valid = Y_valid[:1000]

train_dataset = BaseDataGeneration(X_train, Y_train, args.batch, n_party=args.nparty)
valid_dataset = BaseDataGeneration(X_valid, Y_valid, args.batch, n_party=args.nparty)
auxil_dataset = BaseDataGeneration(X_auxil, Y_auxil, args.batch, n_party=args.nparty)
Y_train_attackers = {}

with strategy.scope():
    model, attackerClassifiers = model_factory(backbone = args.backbone,
                                               n_attackers = args.n_attackers,
                                               n_party = args.nparty)
    
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum),
                  loss = {'output': tf.keras.losses.CategoricalCrossentropy()},
                  metrics = {"output": [tf.keras.metrics.CategoricalAccuracy()]})

    for i in range(len(attackerClassifiers)):
        attackerClassifiers[i].compile(optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum),
                                       loss = {'output': tf.keras.losses.CategoricalCrossentropy()},
                                       metrics = {"output": [tf.keras.metrics.CategoricalAccuracy()]})

positions = {}
for i in range(len(attackerClassifiers)):
    attackerClassifiers[i].fit(auxil_dataset, validation_data = train_dataset, verbose = 1, epochs = args.epochs)
    saliency_maps = compute_saliency_map(train_dataset, attackerClassifiers[i], i)
    positions[i] = np.array([getMaxWindow(saliency_maps[i], args.windowSize)[0] for i in trange(len(saliency_maps))])
    Y_train_attackers[i] = attackerClassifiers[i].predict(train_dataset, verbose=False)

if args.selection:
    targetClass, sourceClass, _ = optimalSelection(model, X_train, Y_train_attackers, list(range(args.n_attackers)), args.nparty, strategy, args.batch, len(set(np.argmax(Y_train, axis=1))))
else:
    targetClass, sourceClass = random.sample(list(range(len(set(Y_train)))), 2)

print("targetClass:", targetClass)
print("sourceClass:", sourceClass)

train_attackDataGeneration = AttackDataGeneration(model, args.p, X_train, Y_train, Y_train_attackers, positions, targetClass, sourceClass, args.windowSize, 
                                                  args.batch, strategy, args.lr, args.momentum, args.epochs, n_party=args.nparty)
detectCallback = DetectCallback(train_attackDataGeneration, np.argmax(Y_train, axis=1), args.nparty)
H = model.fit(train_attackDataGeneration,
              validation_data = valid_dataset,
              callbacks = [detectCallback],
              verbose = 1,
              epochs = args.epochs)

if detectCallback.isAttacked:
    model = unlearning(detectCallback.goodModel)

train_attackDataGeneration.on_epoch_end()
validASRDataGeneration = ASRDataGeneration(X_valid[np.where(np.argmax(Y_valid, axis=1)==sourceClass)[0]], Y_valid[np.where(np.argmax(Y_valid, axis=1)==sourceClass)[0]], 
                                           positions, train_attackDataGeneration.triggers, args.batch, args.nparty)

yPred = model.predict(valid_dataset)
yPred = np.argmax(yPred, axis=1)
yPredASR = model.predict(validASRDataGeneration)
yPredASR = np.argmax(yPredASR, axis=1)
YValid = np.argmax(Y_valid, axis=1)
print("MTA:", np.mean(yPred == YValid))
print("ASR:", np.mean(yPredASR == targetClass), Counter(yPredASR))
print("Predict:", Counter(yPred))