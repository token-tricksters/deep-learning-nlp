from bert import BertModel
import torch
import torch.nn as nn

"""
🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️

This model poses to be a bert model to the outside
but is actually four bert models in disguise!
3 models that sHoUlD be used once per task
1 model that serves as the "gate" function
https://pub.towardsai.net/gpt-4-8-models-in-one-the-secret-is-out-e3d16fd1eee0

🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️🕵️
"""
def softmax_to_onehot(softmax_output):
    max_indices = torch.argmax(softmax_output, dim=-1)
    one_hot = torch.zeros_like(softmax_output)
    one_hot.scatter_(-1, max_indices.unsqueeze(-1), 1)
    return one_hot



class MixtureOfExperts(nn.Module):
    def __init__(self):
        super(MixtureOfExperts, self).__init__()
        self.modela = BertModel.from_pretrained('bert-base-uncased')
        self.modelb = BertModel.from_pretrained('bert-base-uncased')
        self.modelc = BertModel.from_pretrained('bert-base-uncased')

        self.gating_network = BertModel.from_pretrained('bert-base-uncased')
        self.gating_linear_layer = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        # Get the gating probabilities
        gating_output = self.gating_network(input_ids, attention_mask)["pooler_output"]
        gating_probs = self.gating_linear_layer(gating_output)
        gate_probs = torch.nn.functional.softmax(gating_probs, dim=1)

        gate_probs = softmax_to_onehot(gate_probs)


        # Get predictions from each expert
        out_a = self.modela(input_ids, attention_mask)
        out_b = self.modelb(input_ids, attention_mask)
        out_c = self.modelc(input_ids, attention_mask)

        # Combine the predictions
        pooler_output = gate_probs[:, :1] * out_a["pooler_output"] + gate_probs[:, 1:2] * out_b["pooler_output"] + gate_probs[:, 2:3] * out_c["pooler_output"]
        last_hidden_state = gate_probs[:, :1, None] * out_a["last_hidden_state"] + gate_probs[:, 1:2, None] * out_b["last_hidden_state"] + gate_probs[:, 2:3, None] * out_c["last_hidden_state"]


        return {"pooler_output": pooler_output, "last_hidden_state": last_hidden_state}



