from typing import Dict

from tensorflow.keras.callbacks import Callback
from seqeval.metrics import f1_score, classification_report
import numpy as np


class F1scoreKeras(Callback):

    def __init__(self, val_dataset, output_sequence_label_info: Dict, validation_steps=None):
        super(F1scoreKeras, self).__init__()
        self.before_score = 0.0
        self.val_dataset = val_dataset

        self.f1scorer = SparseCategoricalF1Score(output_sequence_label_info)

        self.validation_steps = validation_steps

    def on_epoch_end(self, epoch, logs):
        # 에포크가 끝날 때마다 실행되는 함수

        f1score = self.__validation_f1score()

        logs["val_f1score"] = f1score

    def on_predict_batch_end(self, epoch, logs):
        # tf.keras.Model에 compile에 steps_per_execution 인수가 N으로 설정된 경우 이 메서드는 N batch마다 호출됩니다.

        f1score = self.__validation_f1score()

        logs["val_f1score"] = f1score


    def __validation_f1score(self):
        val_dataset = self.val_dataset

        y_predicteds =[]
        y_val_datas = []

        validation_steps = self.validation_steps
        step_count = 0
        for val_data in val_dataset:
            batch_y_predicted = self.model.predict(val_data[0])
            y_predicteds.append(batch_y_predicted)

            batch_y_val_data = np.array(val_data[1])
            y_val_datas.append(batch_y_val_data)

            step_count += 1
            if validation_steps and step_count == validation_steps:
                break

        y_predicted = np.concatenate(y_predicteds)
        y_val_data = np.concatenate(y_val_datas)

        f1score_report = self.f1scorer.create_report(y_predicted, y_val_data, output_dict=True)
        avg_scores = f1score_report["weighted avg"]
        f1score = avg_scores['f1-score']

        print(f'f1_score improved from {self.before_score} to {f1score}')
        print(' - f1: {:04.2f}'.format(f1score * 100))

        self.before_score = f1score

        return f1score

class SparseCategoricalF1Score:

    def __init__(self, sequence_label_info: Dict):
        self.sequence_label_info = sequence_label_info

    def scoring(self, y_predicted, y_label):

        predict_tags = self.sequences_to_tags_predict(y_predicted)
        label_tags = self.sequences_to_tags_label(y_label)

        score = f1_score(predict_tags, label_tags)

        return score

    def create_report(self, y_predicted, y_label, output_dict=False):
        predict_tags = self.sequences_to_tags_predict(y_predicted)
        label_tags = self.sequences_to_tags_label(y_label)
        report = classification_report(label_tags, predict_tags, output_dict=output_dict)
        return report

    def sequences_to_tags_label(self, sequences_of_data):
        # 예측값을 index_to_ner를 사용하여 태깅 정보로 변경하는 함수.
        sequence_label_info = self.sequence_label_info
        result = []
        for sequences in sequences_of_data:  # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
            tag = []
            for sequence in sequences:  # 시퀀스로부터 예측값을 하나씩 꺼낸다.
                tag.append(sequence_label_info[sequence].replace("PAD", "O"))  # 'PAD'는 'O'로 변경
            result.append(tag)
        return result

    def sequences_to_tags_predict(self, sequences_of_data):
        # 예측값을 index_to_ner를 사용하여 태깅 정보로 변경하는 함수.
        sequence_label_info = self.sequence_label_info
        result = []
        for sequences in sequences_of_data:  # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
            tag = []
            for sequence_one_hot in sequences:  # 시퀀스로부터 예측값을 하나씩 꺼낸다.
                sequence = np.argmax(sequence_one_hot)  # 예를 들어 [0, 0, 1, 0 ,0]라면 1의 인덱스인 2를 리턴한다.
                tag.append(sequence_label_info[sequence].replace("PAD", "O"))  # 'PAD'는 'O'로 변경
            result.append(tag)
        return result
