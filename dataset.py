from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import csv


class ChessDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.sequences = []
        with open(csv_file, "r") as file:
            csv_reader = csv.reader(file)
        for row in csv_reader:
            self.sequences.append(row)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        sequence_tokens = [move for move in sequence]
        input_ids = self.tokenizer.convert_tokens_to_ids(sequence_tokens)
        return input_ids


def collate_fn(batch):
    input_ids = [torch.tensor(ids) for ids in batch]
    padded_input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=0)
    attention_mask = (padded_input_ids != 0).float()
    return padded_input_ids, attention_mask
