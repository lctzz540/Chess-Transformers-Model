import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from dataset import ChessDataset, collate_fn
from network import ChessTransformer
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="ChessTransformer")
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 32)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=14,
    metavar="N",
    help="number of epochs to train (default: 14)",
)

parser.add_argument(
    "--lr",
    type=float,
    default=1e-5,
    metavar="LR",
    help="learning rate (default: 1e-5)",
)


parser.add_argument("--device", default="cpu", help="choose device")


args = parser.parse_args()


model_name = "bert-base-uncased"
chess_transformer = ChessTransformer(model_name)

BATCH_SIZE = args.batch_size
device = torch.device(args.device)


dataset = ChessDataset(
    csv_file="./moves_history.csv",
    tokenizer=chess_transformer.tokenizer,
)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

chess_transformer.to(device)

optimizer = AdamW(chess_transformer.parameters(), lr=args.lr)
best_loss = float("inf")
epochs_without_improvement = 0
patience = 3

loss_history = []

for epoch in range(20):
    print(f"Epoch {epoch+1}")
    progress_bar = tqdm(total=len(dataloader))
    for batch in dataloader:
        input_ids, attention_mask = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        logits = chess_transformer((input_ids, attention_mask))

        logits = logits.view(-1, logits.shape[-1])
        target_labels = input_ids.view(-1)
        target_labels = target_labels[: logits.shape[0]]

        loss = torch.nn.CrossEntropyLoss()(logits, target_labels.view(-1))
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update(1)
        if progress_bar.n % progress_bar.total == 0:
            progress_bar.refresh()

    loss_history.append(loss.item())
    progress_bar.close()

    if loss < best_loss:
        best_loss = loss
        torch.save(chess_transformer.state_dict(), "best_weights.pth")
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement == patience:
        print(
            f"Early stopping triggered. No improvement in loss for {patience} epochs."
        )
        break
    print(f"Epoch {epoch} is done")

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.savefig("loss_graph.png")
plt.show()
