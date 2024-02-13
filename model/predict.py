import os
import pickle
import torch
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm
from model.utils.forward_fn import forward_word_classification

entity_map_dic = {"PLACE": "address", "PERSON": "name", "ORGANISATION": "organisation"}

class PredictLabel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = self._load_model()
    
    def _load_model(self):
        model = torch.load(os.path.join(self.model_path, 'ner_model.ckpt'), map_location='cpu')
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model.eval()
        return model

    def _bio_data_handler(self, sentence, predict_label):
        entities = []
        pre_label = predict_label[0]
        word = ""
        for i in range(len(sentence)):
            current_label = predict_label[i]
            if current_label.startswith('B'):
                if pre_label[2:] is not current_label[2:] and word != "":
                    entities.append([word, entity_map_dic[pre_label[2:]]])
                    word = ""
                pre_label = current_label
                word += sentence[i]
            elif current_label.startswith('I'):
                word += ' '
                word += sentence[i]
                pre_label = current_label
            elif current_label.startswith('O'):
                if pre_label[2:] is not current_label[2:] and word != "":
                    entities.append([word, entity_map_dic[pre_label[2:]]])
                pre_label = current_label
                word = ""
        if word != "":
            entities.append([word, entity_map_dic[pre_label[2:]]])
        return entities

    def _combine_person_entity(self, entity):
        person_types = ['first_name', 'last_name']
        sort_entity = sorted(entity, key=lambda x: x['begin'])
        person_type_node = [e for e in sort_entity if e["type"] in person_types]
        other_type_node = [e for e in sort_entity if e["type"] not in person_types]
        new_entity, res_entity = list(), list()
        if len(person_type_node) == 0:
            return entity

        e1 = person_type_node[0]
        for e2 in person_type_node[1:]:
            if self._check_entity_continuous(e1, e2):
                ent = self._merge_entity(e1, e2)
                e1 = ent
            else:
                new_entity.append(e1)
                e1 = e2
        new_entity.append(e1)

        for e in new_entity:
            if e["type"] in person_types:
                e_info = {'type': 'person', 'value': e["value"], 'begin': e["begin"],
                          'end': e["end"], 'detail': [e]}
                res_entity.append(e_info)
            else:
                res_entity.append(e)
        res_entity.extend(other_type_node)
        return res_entity
    
    def _check_entity_continuous(self, e1, e2):
        return True if e1['end'] == e2['begin'] else False

    def _merge_entity(self, e1, e2):
        r_entity = dict()
        person_types = ['first_name', 'last_name', 'person']
        r_entity['type'] = 'person' if e1['type'] in person_types else e1['type']
        r_entity['value'] = e1['value'] + e2['value']
        r_entity['begin'] = e1['begin']
        r_entity['end'] = e2['end']
        detail = e1.get('detail', None)
        if detail:
            detail.append(e2)
            r_entity['detail'] = detail
        else:
            r_entity['detail'] = [e1, e2]
        return r_entity

    def get_entities_result(self, query):
        sentence_list, predict_labels =self.predict(query)

        if len(predict_labels) == 0:
            print("句子: {0}\t实体识别结果为空".format(query))
            return []

        entities = []
        if len(sentence_list) == len(predict_labels):
            result = self._bio_data_handler(sentence_list, predict_labels)
            # print(result)
            if len(result) != 0:
                end = 0
                prefix_len = 0

                for word, label in result:
                    sen = query[end:]
                    begin = sen.find(word) + prefix_len
                    end = begin + len(word)
                    prefix_len = end
                    if begin != -1:
                        ent = dict(value=query[begin:end], type=label)
                        entities.append(ent)
        return entities

    def predict(self, sentence):
        dataset = []
        sentence = sentence.split()
        length = len(sentence)
        seq_label = [5] * length
        dataset.append({
                        'sentence': sentence,
                        'seq_label': seq_label
                    })

        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS

        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
        
        
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1] 

        batch = [(np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), sentence)]

        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(512, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))

        batch_size = len(batch)

        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)

        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            # (subword_batch, mask_batch, subword_to_word_indices_batch, label_batch) = batch
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)

        batch_data = (np.array(subword_batch), np.array(mask_batch), np.array(subword_to_word_indices_batch), np.array(seq_label_batch))
        model = torch.load(os.path.join(self.model_path, "ner_model.ckpt"), map_location="cpu")
        
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model.eval()

        total_loss, total_correct, total_labels = 0, 0, 0
        list_hyp, list_label, list_seq = [], [], []
        
        i2w = {0: 'I-PERSON', 1: 'B-ORGANISATION', 2: 'I-ORGANISATION', 3: 'B-PLACE', 4: 'I-PLACE', 5: 'O', 6: 'B-PERSON'}
        loss, batch_hyp, batch_label = forward_word_classification(self.model, batch_data, i2w=i2w, device='cpu')

        return sentence, batch_hyp[0]

