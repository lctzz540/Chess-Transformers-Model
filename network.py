import torch
from transformers import BertTokenizer, BertModel, AdamW


class ChessTransformer(torch.nn.Module):
    def __init__(self, model_name):
        super(ChessTransformer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.transformer = BertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(
            self.transformer.config.hidden_size, self.tokenizer.vocab_size
        )

    def forward(self, inputs):
        input_ids = inputs[0]
        attention_mask = inputs[1]
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
