import os
import torch
import util.namesgenerator as namesgenerator
from torch.onnx import export
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
from bart_onnx.generation_onnx import BARTBeamSearchGenerator


class OnnxModelConverter:
    def __init__(self, prefix):
        self.prefix = prefix
        self.__tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
    
    def __load_sample_inputs(self, max_length):
        tokenizer = self.__tokenizer
        text = """ 국제축구연맹(FIFA) 월드컵 제1회 대회가 열린 1930년 우루과이를 시작으로 92년의 역사를 지닌 남자 월드컵에서 여성 심판이 주심을 맡은 건 이날이 처음이다. 
        이날 경기는 주심뿐 아니라 부심까지 모두 여성으로 꾸려졌다. 브라질의 네우사 백 심판과 멕시코 카렌 디아스 심판이 프라파르 심판과 함께 그라운드에 나섰다. 
        오프사이드 비디오판독(VAR) 임무 역시 또 다른 여성 심판 캐스린 네즈빗(미국)이 맡았다. """
        text = text.replace('\n', ' ')

        inputs = tokenizer([text], max_length=max_length, return_tensors="pt")

        return inputs

    def convert(self):

        model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
        bart_script_model = torch.jit.script(BARTBeamSearchGenerator(model))
        
        model_onnx_path = os.path.join(self.prefix, 'onnx')

        num_beams=4
        max_length=512
        eos_token_id=1
        inputs = self.__load_sample_inputs(max_length)
        
        if os.path.exists(model_onnx_path):
            file_name = 'model.onnx'
            random_file_path = os.path.join(model_onnx_path, namesgenerator.get_random_name())
            os.mkdir(random_file_path)
            model_onnx_path = os.path.join(random_file_path, file_name)
            print(f"==file exists, model saved path gonna be changed to '{model_onnx_path}'")
        else:
            os.mkdir(model_onnx_path)
            file_name = 'model.onnx'
            model_onnx_path = os.path.join(model_onnx_path, file_name)

        # summary_ids = model.generate(
        #     inputs["input_ids"],
        #     attention_mask=inputs["attention_mask"],
        #     num_beams=torch.tensor(num_beams),
        #     max_length=torch.tensor(max_length),
        #     early_stopping=True,
        #     eos_token_id=torch.tensor(eos_token_id)
        # )

        export(
            bart_script_model,
            (
                inputs["input_ids"],
                inputs["attention_mask"],
                torch.tensor(num_beams),
                torch.tensor(max_length),
                torch.tensor(eos_token_id),
            ),
            model_onnx_path,
            opset_version=13,
            input_names=["input_ids", "attention_mask", "num_beams", "max_length", "decoder_start_token_id"],
            output_names=["summary_ids"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "summary_ids": {0: "batch", 1: "seq_out"},
            },
        )

