# -*- coding: utf-8 -*-
# @description: 
# @file: config.py

import datetime
import os
import threading
from utils.enums import STRATEGY


class Config(object):
    _instance_lock = threading.Lock()#多线程锁
    _init_flag = False

    def __init__(self):
        if not Config._init_flag:
            Config._init_flag = True
            root_path = str(os.getcwd()).replace("\\", "/")
            if 'source' in root_path.split('/'):
                self.base_path = os.path.abspath(os.path.join(os.path.pardir))
            else:
                self.base_path = os.path.abspath(os.path.join(os.getcwd()))
            self._init_train_config()

    def __new__(cls, *args, **kwargs):
        if not hasattr(Config, '_instance'):
            # https://zhuanlan.zhihu.com/p/360604465
            with Config._instance_lock:
                if not hasattr(Config, '_instance'):
                    Config._instance = object.__new__(cls)
        return Config._instance

    def _init_train_config(self):
        self.label_list = []
        self.use_gpu = True
        self.device = "cpu"
        self.sep = "\t" #separator in train/test/eval txt

        # 输入数据集、输出目录
        self.train_file = os.path.join(self.base_path, 'dataset/nergrit_ner-grit', 'train_preprocess.txt')
        self.eval_file = os.path.join(self.base_path, 'dataset/nergrit_ner-grit', 'valid_preprocess.txt')
        self.test_file = os.path.join(self.base_path, 'dataset/nergrit_ner-grit', 'test_preprocess.txt')
        self.log_path = os.path.join(self.base_path, 'save', "logs")
        self.output_path = os.path.join(self.base_path, 'save', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

        # self.model_name_or_path = 'indobenchmark/indobert-large-p2'
        self.model_name_or_path = 'indolem/indobert-base-uncased'
        # self.model_name_or_path = 'bert-base-multilingual-cased'
        # self.model_name_or_path = 'denaya/indoSBERT-large'


        # 以下是模型训练参数
        self.do_train = True
        self.do_eval = True
        self.do_test = False
        self.clean = True
        self.need_birnn = True
        self.do_lower_case = True
        self.rnn_dim = 128
        self.max_seq_length = 128
        self.train_batch_size = 4
        self.eval_batch_size = 2
        self.num_train_epochs = 20
        self.gradient_accumulation_steps = 5
        self.learning_rate = 3e-5
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.logging_steps = 500


class ModelConfig(object):
    def __init__(self):
        self.embed_dim = 300
        self.dropout = 0.5
        self.lstm_size = 256
        self.lstm_layer = 1
        self.vocab_size = 30000
        self.use_pretrained = True
        self.num_oov_buckets = 1
        self.embed = './data/chinese/sgn_renmin.npy'
        self.vocab = './data/chinese/vocab.txt'
        self.positive_tags = ['B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-BOOK', 'I-BOOK', 'B-COMP',
                              'I-COMP', 'B-GAME', 'I-GAME', 'B-GOVERN', 'I-GOVERN', 'B-MOVIE', 'I-MOVIE', 'B-POS',
                              'I-POS', 'B-SCENE', 'I-SCENE']
        self.positive_ids = None
        self.tags = None
        self.buffer_size = 15000
        self.epochs = 20
        self.batch_size = 32
        self.save_checkpoints_steps = 500
        self.model_dir = './results/model_chinese'
        self.update()

    def update(self):
        """
        When you reset any parameters, call this method to update the relevant parameters.
        """
        self.positive_ids = list(range(len(self.positive_tags)))
        self.tags = self.positive_tags + ['O', 'X', '[CLS]', '[SEP]']


class ActiveConfig(object):
    def __init__(self, pool_size, total_percent=0.5, select_percent=0.25, select_epochs=10, total_epochs=20):
        self.pool_size = pool_size
        self.total_percent = total_percent
        self.select_percent = select_percent
        self.total_num = None
        self.select_num = None
        self.select_strategy = STRATEGY.MNLP
        self.select_epochs = select_epochs  # the train epochs each time new samples are added
        self.total_epochs = total_epochs  # the train epochs from scratch when finish sampling
        self.update()

    def update(self):
        """
        When you reset any parameters, call this method to update the relevant parameters.
        """
        self.total_num = int(self.pool_size*self.total_percent)
        self.select_num = int(self.total_num*self.select_percent)
