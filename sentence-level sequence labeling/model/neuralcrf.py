import torch
import torch.nn as nn

from crf import LinearCRF
from encoder import TransformerEncoder


class NNCRF(nn.Module):

    def __init__(self, config):
        super(NNCRF, self).__init__()
        self.device = config.device
        self.encoder = TransformerEncoder(config)
        self.inferencer = LinearCRF(config)

    def forward(self, doc_tensor, attention_mask_tensor, cls_tensor, doc_len_tensor, label_tensor):
        # Encode.
        _, lstm_scores = self.encoder(doc_tensor, attention_mask_tensor, cls_tensor)

        # Inference.
        batch_size = cls_tensor.size(0)
        sent_len = cls_tensor.size(1)
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len).to(self.device)
        mask = torch.le(maskTemp, doc_len_tensor.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)
        unlabed_score, labeled_score = self.inferencer(lstm_scores, doc_len_tensor, label_tensor, mask)
        return unlabed_score - labeled_score

    def decode(self, batchInput):
        """
        Decode the batch input
        """
        doc_tensor, attention_mask_tensor, cls_tensor, doc_len_tensor, label_tensor = batchInput
        # Encode.
        feature_out, features = self.encoder(doc_tensor, attention_mask_tensor, cls_tensor)
        # Decode.
        bestScores, decodeIdx = self.inferencer.decode(features, doc_len_tensor)
        return bestScores, decodeIdx