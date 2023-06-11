import torch
from transformers import BertTokenizer

from chess_transformer import ChessTransformer

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = ChessTransformer(model_name)

model.load_state_dict(torch.load("best_weights.pth"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def is_valid_move(predicted_move):
    return True


input_moves = ["e4", "d5"]

while True:
    input_tokens = tokenizer.tokenize(" ".join(input_moves))
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model((input_ids_tensor, torch.ones_like(input_ids_tensor)))
        predicted_next_move_id = torch.argmax(logits).item()
        predicted_next_move_token = tokenizer.convert_ids_to_tokens(
            predicted_next_move_id
        )

    if is_valid_move(predicted_next_move_token):
        break

    input_moves.append(predicted_next_move_token)

print("Predicted next move:", predicted_next_move_token)
