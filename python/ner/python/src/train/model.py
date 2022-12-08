import tensorflow as tf
import pickle
import json
import os
from tqdm import tqdm
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import wandb
from train.callback.f1score import F1scoreKeras
from wandb.keras import WandbCallback
from util.utils import (
    TOKENIZER_CLASSES,
    MODEL_FOR_TOKEN_CLASSIFICATION
)
import logging
logging.basicConfig(level = logging.INFO)


class RawDataToTrainingDataConverter:

    def __init__(self, file_path, model_type, model_name):
        self.model_type = model_type
        self.model_name = model_name
        with open(file_path, encoding='utf-8-sig') as f:
            self.raw_data = json.load(f)['document']

    def convert(self):
        label_dict, index_to_ner = self.__make_label_dictionary()
        logging.info("label dictionary created..")
        logging.info(label_dict)
        train_dataset = self.__make_featured_arr(label_dict)
        logging.info("train dataset created..")
        logging.info(f"train dataset length : {len(train_dataset['tr_inputs'])}")
        logging.info(f"validation dataset length : {len(train_dataset['val_inputs'])}")
        return train_dataset, label_dict

    def __make_label_dictionary(self):

        label_arr = []
        for document in self.raw_data:
            for sentence in document['sentence']:
                for ne in sentence['ne']:
                    label_arr.append(ne['label'])

        unique_label_arr = set(label_arr)

        all_label_arr = []
        for label in unique_label_arr:
            all_label_arr.append(label + '_B')
            all_label_arr.append(label + '_I')

        # '-' , 'PAD' 토큰 추가
        all_label_arr.insert(0, '-')
        all_label_arr.insert(0, 'PAD')

        label_dict = {word: i for i, word in enumerate(all_label_arr)}
        index_to_ner = {i: j for j, i in label_dict.items()}

        return label_dict, index_to_ner

    def __make_featured_arr(self, label_dict):

        featurer = NERDataFeaturing(self.model_type, self.model_name, label_dict)

        input_ids_arr = []
        attention_mask_arr = []
        output_ids_arr = []
        for doc in tqdm(self.raw_data):
            for sentence in doc['sentence']:
                input_ids, attention_mask, token_type_ids, output_ids_pad = featurer.featuring(sentence)
                input_ids_arr.append(input_ids)
                attention_mask_arr.append(attention_mask)
                output_ids_arr.append(output_ids_pad)

        input_ids_arr = np.array(input_ids_arr)
        attention_mask_arr = np.array(attention_mask_arr)
        output_ids_arr = np.array(output_ids_arr)

        tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids_arr, output_ids_arr,
                                                                    random_state=2018, test_size=0.1)
        tr_masks, val_masks, _, _ = train_test_split(attention_mask_arr, input_ids_arr,
                                                     random_state=2018, test_size=0.1)
        train_dataset = {
            'tr_inputs' : tr_inputs,
            'val_inputs': val_inputs,
            'tr_masks' : tr_masks,
            'val_masks': val_masks,
            'tr_tags': tr_tags,
            'val_tags': val_tags
        }
        return train_dataset

class NERDataFeaturing:

    def __init__(self, model_type, model_name, label_dict):
        self.__model_type = model_type
        self.__tokenizer = TOKENIZER_CLASSES[model_type].from_pretrained(model_name)
        self.__label_sequence_info = label_dict
        self.__max_length = 128

    def featuring(self, sentence_info):
        return self.__parse_sentence_info_to_feature(sentence_info)

    def __parse_sentence_info_to_feature(self, sentence_info):

        sentence_text = sentence_info['form']
        sentence_ne_infos = sentence_info['ne']

        token_list, token_ner_list, token_ner_id_list = self.__tokenize_sentence_and_ne_mapping(sentence_text,
                                                                                                sentence_ne_infos)

        # input featuring
        input_ids, attention_mask, token_type_ids = self.__featuring_model_input(sentence_text)

        # output featuring
        ## 앞에 special token을 채우기
        output_ids = self.__parse_label(token_ner_list)
        output_ids_pad = [1] + output_ids + [1]
        output_ids_pad = pad_sequences([output_ids_pad], value=self.__label_sequence_info["PAD"],
                                       maxlen=self.__max_length, padding='post',
                                       truncating='post')

        output_ids_pad = output_ids_pad[0].tolist()
        output_ids = np.array(output_ids_pad)

        return input_ids, attention_mask, token_type_ids, output_ids

    def __featuring_model_input(self, sentence_text):

        model_input = self.__tokenizer(
            sentence_text,
            return_tensors='np',
            truncation=True,
            max_length=self.__max_length,
            padding="max_length",
            add_special_tokens=True
        )
        input_ids = model_input["input_ids"][0]
        attention_mask = model_input["attention_mask"][0]
        token_type_ids = model_input["token_type_ids"][0]

        return input_ids, attention_mask, token_type_ids

    def __parse_label(self, labels):
        label_sequence_info = self.__label_sequence_info
        sequences = [label_sequence_info[label] for label in labels]
        return sequences

    def __tokenize_sentence_and_ne_mapping(self, sentence_text, sentence_ne_infos):
        # char 별 ne 정보를 사전화
        id_ne_vocab = {}
        char_idx_tag_dict = {}

        for sentence_ne_info in sentence_ne_infos:

            id_ne_vocab[sentence_ne_info['id']] = sentence_ne_info

            for char_idx in range(sentence_ne_info['begin'], sentence_ne_info['end']):
                char_idx_tag_dict[char_idx] = sentence_ne_info

        # 토큰별 ne id 정보를 mapping
        tokenizer = self.__tokenizer
        token_list = []
        token_ner_id_list = []

        s_char_idx = 0
        for word in sentence_text.split(' '):
            word_token_list = tokenizer.tokenize(word)
            token_list.extend(word_token_list)

            for token in word_token_list:

                try:
                    char_tag = char_idx_tag_dict[s_char_idx]
                    tag_id = char_tag['id']
                except KeyError:
                    tag_id = 0

                token_ner_id_list.append(tag_id)

                if('kobert' in self.__model_type):
                    split_char = '_'
                else:
                    split_char = '##'
                s_char_idx += len(token.replace(split_char, ''))

            # 공백 idx
            s_char_idx += 1

        # 토큰 label mapping
        token_ner_list = []
        before_id = 0
        for token_ner_id in token_ner_id_list:

            if token_ner_id == 0:
                token_ner_list.append('-')
                continue

            id_tag = id_ne_vocab[token_ner_id]
            label = id_tag['label']

            if before_id == id_tag['id']:
                label = label + '_I'
            else:
                label = label + '_B'

            token_ner_list.append(label)

            before_id = id_tag['id']

        return token_list, token_ner_list, token_ner_id_list

class NERClassificationModel:

    def __init__(self):
        self.SEQ_LEN = 128

    def create_model(self, model_type, model_name, num_labels):
        SEQ_LEN = self.SEQ_LEN
        model = MODEL_FOR_TOKEN_CLASSIFICATION[model_type].from_pretrained(model_name, from_pt=True, num_labels=num_labels,
                                            output_attentions=False, output_hidden_states=False)
        token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')  # 토큰 인풋
        mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')  # 마스크 인풋
        bert_outputs = model([token_inputs, mask_inputs])
        bert_outputs = bert_outputs[0]  # shape : (Batch_size, max_len, num_labels(개체의 총 개수))
        nr = tf.keras.layers.Dense(num_labels, activation='softmax')(
            bert_outputs)  # shape : (Batch_size, max_len, num_labels)
        nr_model = tf.keras.Model([token_inputs, mask_inputs], nr)

        nr_model.compile(optimizer=tf.keras.optimizers.Adam(lr=5e-5),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                         metrics=['sparse_categorical_accuracy'])
        nr_model.summary()
        return nr_model

    def train(self, args):

        corpus_path = args.corpus_path
        export_path = args.export_path
        model_type = args.model_type
        model_name = args.model_name
        num_epoch = args.num_epoch

        export_model_path = os.path.join(export_path, str(model_name).replace("/","-"))

        wandb.login()
        wandb.init(project=f'{model_type}')

        if os.path.exists(os.path.join(export_model_path,'data')):
            train_dataset = {}
            with open(os.path.join(export_model_path, 'data/label_dict.pkl'), 'rb') as f:
                label_dict = pickle.load(f)
            with open(os.path.join(export_model_path, 'data/tr_inputs.pkl'), 'rb') as f:
                train_dataset['tr_inputs'] = pickle.load(f)
            with open(os.path.join(export_model_path, 'data/tr_masks.pkl'), 'rb') as f:
                train_dataset['tr_masks'] = pickle.load(f)
            with open(os.path.join(export_model_path, 'data/tr_tags.pkl'), 'rb') as f:
                train_dataset['tr_tags'] = pickle.load(f)

            with open(os.path.join(export_model_path, 'data/val_inputs.pkl'), 'rb') as f:
                train_dataset['val_inputs'] = pickle.load(f)
            with open(os.path.join(export_model_path, 'data/val_masks.pkl'), 'rb') as f:
                train_dataset['val_masks'] = pickle.load(f)
            with open(os.path.join(export_model_path, 'data/val_tags.pkl'), 'rb') as f:
                train_dataset['val_tags'] = pickle.load(f)
        else:
            os.makedirs(export_model_path)
            os.makedirs(export_model_path + '/data')
            raw_data_converter = RawDataToTrainingDataConverter(corpus_path, model_type, model_name)
            train_dataset, label_dict = raw_data_converter.convert()

            with open(os.path.join(export_model_path, 'data/label_dict.pkl'), 'wb') as f:
                pickle.dump(label_dict, f)
            with open(os.path.join(export_model_path, 'data/tr_inputs.pkl'), 'wb') as f:
                pickle.dump(train_dataset['tr_inputs'], f)
            with open(os.path.join(export_model_path, 'data/tr_masks.pkl'), 'wb') as f:
                pickle.dump(train_dataset['tr_masks'], f)
            with open(os.path.join(export_model_path, 'data/tr_tags.pkl'), 'wb') as f:
                pickle.dump(train_dataset['tr_tags'], f)

            with open(os.path.join(export_model_path, 'data/val_inputs.pkl'), 'wb') as f:
                pickle.dump(train_dataset['val_inputs'], f)
            with open(os.path.join(export_model_path, 'data/val_masks.pkl'), 'wb') as f:
                pickle.dump(train_dataset['val_masks'], f)
            with open(os.path.join(export_model_path, 'data/val_tags.pkl'), 'wb') as f:
                pickle.dump(train_dataset['val_tags'], f)

        nr_model = self.create_model(model_type, model_name, len(label_dict))

        validation_steps = int(num_epoch / 10) if num_epoch else None
        fl_score_callback = F1scoreKeras(
            [(train_dataset['val_inputs'], train_dataset['val_masks']), train_dataset['val_tags']],
            label_dict,
            validation_steps
        )
        nr_model.fit([train_dataset['tr_inputs'], train_dataset['tr_masks']], train_dataset['tr_tags']
                     , validation_data=([train_dataset['val_inputs'], train_dataset['val_masks']], train_dataset['val_tags']), epochs=num_epoch,
                     shuffle=False, batch_size=32
                     ,callbacks=[fl_score_callback, WandbCallback(save_model=False)])
        nr_model.save_weights(os.path.join(export_model_path, "model.h5"))
