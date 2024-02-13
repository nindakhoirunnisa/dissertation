import logging
import math
import os
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import logit, nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers import AlbertPreTrainedModel, BertPreTrainedModel, XLMPreTrainedModel, AlbertModel, BertModel, BertConfig, XLMModel, XLMConfig, XLMRobertaModel, XLMRobertaConfig
from transformers import AutoTokenizer, AutoConfig

from torchcrf import CRF

class BertForWordClassification(BertPreTrainedModel):
  def __init__(self, config, need_birnn=True, rnn_dim=128):
  # def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels

    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # self.classifier = nn.Linear(config.hidden_size, config.num_labels) #BERT only

    #add BiLSTM-CRF
    # out_dim = config.hidden_size
    # if need_birnn:
    #   self.need_birnn = need_birnn
    #   self.birnn = nn.LSTM(input_size=config.hidden_size, hidden_size=rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
    #   out_dim = rnn_dim * 2
    # self.dropout_after = nn.Dropout(config.hidden_dropout_prob)
    # self.hidden2tag = nn.Linear(in_features=out_dim, out_features=config.num_labels)
    self.hidden2tag = nn.Linear(config.hidden_size, out_features=config.num_labels)
    self.crf = CRF(num_tags=config.num_labels, batch_first=True)
    #End add BiLSTM-CRF

    self.init_weights()
  
  def forward(
    self, 
    input_ids=None,
    subword_to_word_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None
    ):
    outputs = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
    )

    sequence_output = outputs[0] #why index=0?

    ## sequence_output = self.dropout(sequence_output)

    max_seq_len = subword_to_word_ids.max() + 1
    word_latents = []
    for i in range(max_seq_len):
      mask = (subword_to_word_ids == i).unsqueeze(dim=-1)
      word_latents.append((sequence_output * mask).sum(dim=1) / mask.sum())
    word_batch = torch.stack(word_latents, dim=1)

    sequence_output = self.dropout(word_batch)

    # sequence_output, _ = self.birnn(word_batch)
    # sequence_output = self.dropout_after(sequence_output)

    logits = self.hidden2tag(sequence_output)

    expanded_attention_mask = attention_mask[:, :word_batch.shape[1]]

    mask = expanded_attention_mask.unsqueeze(2)
    crf_logits = logits * mask - (1 - mask) * 1e5
    decoded = self.crf.decode(crf_logits)

    # decoded = self.crf.decode(logits, expanded_attention_mask.byte())
    outputs = (decoded, ) + outputs[2:] #return crf 
    if labels is not None:
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      outputs = (loss,) + outputs

    # sequence_output = self.dropout(word_batch)
    # logits = self.classifier(word_batch)
    # outputs = (logits,) + outputs[2:]
    
    # if labels is not None:
    #   loss_fct = CrossEntropyLoss()
    #   loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    #   outputs = (loss,) + outputs
    return outputs #(loss), scores, (hidden_states), (attentions)


if __name__ == '__main__':
  print("BertForWordClassification")
  tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
  config = AutoConfig.from_pretrained("indobenchmark/indobert-base-p1")
  model = BertForWordClassification.from_pretrained("indobenchmark/indobert-base-p1", config=config) 
