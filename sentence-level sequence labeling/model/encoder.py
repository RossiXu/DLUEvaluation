import torch
import torch.nn as nn
from transformers import BertModel


class TransformerEncoder(nn.Module):

    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        # parameters
        self.device = config.device
        self.label_size = config.label_size
        self.input_size = config.embedding_dim
        self.label2idx = config.label2idx
        self.labels = config.idx2labels
        final_hidden_dim = config.hidden_dim

        # model
        self.transformer = BertModel.from_pretrained(config.transformer).to(self.device)
        self.word_drop = nn.Dropout(config.dropout).to(self.device)
        self.linear = nn.Linear(self.input_size, config.hidden_dim).to(self.device)
        self.hidden2tag = nn.Linear(final_hidden_dim, self.label_size).to(self.device)

    def forward(self, doc_tensor, attention_mask_tensor, cls_tensor):
        """
        :param attention_mask_tensor: (bs, max_doc_token_num)
        :param doc_tensor: (bs, max_doc_token_num)
        :param cls_tensor: (bs, max_doc_sent_num)
        :return:
        """
        # get doc embedding
        doc_output = self.transformer(doc_tensor, attention_mask=attention_mask_tensor)[0]  # (bs, max_doc_token_num, embed_dim)

        # get sentence embedding
        batch_size = doc_tensor.shape[0]
        sent_reps = []
        for i in range(batch_size):
            sent_rep = doc_output[i, cls_tensor[i], :]  # (sent_num, embed_dim)
            sent_rep = sent_rep.view(1, sent_rep.size(0), sent_rep.size(1))
            sent_reps.append(sent_rep)
        sent_reps = torch.cat(sent_reps)  # (bs, max_doc_sent_num, embed_dim)

        # linear func
        sent_rep = self.word_drop(sent_reps)
        feature_out = self.linear(sent_rep)
        outputs = self.hidden2tag(feature_out)

        return feature_out, outputs

