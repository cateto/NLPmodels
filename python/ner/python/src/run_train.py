import argparse
from train.model import NERClassificationModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model_type", help="모델 종류 ex) kobert, distilkobert, koelectra-base-v3, roberta-large", required=True
    )
    parser.add_argument(
        "--model_name", help="모델 이름. huggingface에 등록된 이름", required=True
    )

    parser.add_argument(
        "--corpus_path", help="학습데이터 json 파일 경로", required=True
    )

    parser.add_argument(
        "--export_path", help="모델 및 학습데이터 내보내는 경로", required=True
    )
    parser.add_argument(
        "--num_epoch", help="epoch 수", default=3
    )
    args, _ = parser.parse_known_args()

    # corpus_file_path = '/repository/NER/data/k_corpus/json_data/corpus_v2.json'
    # export_path = '/repository/som/'

    model = NERClassificationModel()
    model.train(args)

