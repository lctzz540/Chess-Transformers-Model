# Chess Transformers Model

## Installation

Git Clone this repository

```bash
git clone https://github.com/lctzz540/Chess-Transformers-Model.git
```

## Get Data for trainning

Dataset for trainning this model is the record of moves history of winning match
By running following commands, more data will be add to `moves_history.csv`

```bash
go get
go run .

```

## Training

Traning on default:

```bash
python train.py

```

> usage: train.py [-h] [--batch-size N] [--epochs N] [--lr LR] [--device DEVICE]
> ChessTransformer
>
> options:
> -h, --help show this help message and exit
> --batch-size N input batch size for training (default: 32)
> --epochs N number of epochs to train (default: 14)
> --lr LR learning rate (default: 1e-5)
> --device DEVICE choose device

## Traning weight

Link to training weight: <https://drive.google.com/file/d/1-8IuqbKeMrO2tr3TWfrLsyDWbxJK8xYB/view?usp=sharing>
