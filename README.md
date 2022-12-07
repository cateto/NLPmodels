# NLPmodels
NLPmodels to USE


## NER Trainer 
### Local
```shell
cd python/ner/python/src
pip install -r requirements.txt
python run_train.py --model_type [모델 타입] --model_name [모델이름] --corpus_path [코퍼스json위치] --export_path [모델 및 학습 데이터 저장 경로]
```

### Docker container
```shell
docker build . -t lab-ner
docker run -v [데이터가 있는 로컬 경로]:/repository lab-ner
```
