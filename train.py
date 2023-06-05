import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from torchsummary import summary
from dataset import ChessDataset, collate_fn
from network import ChessTransformer
import argparse

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

for epoch in range(args.epochs):
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

# Test inference
test_input = ["e4", "d5", "exd5", "Nf6"]
test_input_tokens = chess_transformer.tokenizer.tokenize(" ".join(test_input))
test_input_ids = chess_transformer.tokenizer.convert_tokens_to_ids(
    test_input_tokens)
test_input_ids = torch.tensor(test_input_ids).unsqueeze(0).to(device)
test_attention_mask = torch.ones_like(test_input_ids).to(device)
test_logits = chess_transformer((test_input_ids, test_attention_mask))
predicted_next_move_id = torch.argmax(test_logits).item()
predicted_next_move_token = chess_transformer.tokenizer.convert_ids_to_tokens(
    predicted_next_move_id
)

print("Predicted next move:", predicted_next_move_token)

summary(
    chess_transformer,
    input_size=[
        (BATCH_SIZE, len(test_input_tokens)),
        (BATCH_SIZE, len(test_input_tokens)),
    ],
    col_names=["Input IDs", "Attention Mask"],
).save("model_summary.txt")
