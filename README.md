# VerticalFL

## Install
```console
git clone https://github.com/ffyyytt/FedYOLO.git
pip install requirements.txt
```

## Under no attack
```console
python3 no-attack.py
```
Parameters:
- epochs: Number of epochs (default: 100)
- batch: Batch size (default: 32)
- lr: Learning rate (default: 1e-2)
- momentum: SGD Momentum (default: 0.9)
- nparty: number of clients (default: 2)
- backbone: Backbone model (default: resnet18)

# Attack
```console
python3 attack.py
```
Parameters:
- epochs: Number of epochs (default: 100)
- batch: Batch size (default: 32)
- lr: Learning rate (default: 1e-2)
- momentum: SGD Momentum (default: 0.9)
- nparty: number of clients (default: 2)
- backbone: Backbone model (default: resnet18)
- n_attackers: Numer of attackers (default: 1)
- windowSize: Trigger window size (default: 5)
- p: Percentage subset of the source class (default: 0.5)
- selection: Optimal Selection (default: True)
