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

    parser.add_argument(
        "--max_length", help="max_length", default=128
    )
    
    parser.add_argument(
        "--batch_size", help="batch_size", default=32
    )
    args, _ = parser.parse_known_args()

    model = NERClassificationModel()
    model.train(args)

## python run_train.py --model_type kobert --model_name "monologg/kobert" --corpus_path "/repository/NER/data/k_corpus/json_data/corpus_v3.json" --export_path "/repository/som/outputs"

