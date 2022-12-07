from train.prepare_model import OnnxModelConverter

prefix = '/workspace/github/NLPmodels/model'
pmc = OnnxModelConverter(prefix)  # onnx 모델 변환
pmc.convert()
