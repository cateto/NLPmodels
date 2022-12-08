from transformers import (
    ElectraTokenizer,
    XLMRobertaTokenizer,
    AutoTokenizer,
    TFDistilBertForTokenClassification,
    TFElectraForTokenClassification,
    TFBertForTokenClassification,
    TFXLMRobertaForTokenClassification,
    TFAutoModel
)

from train.tokenization_kobert import KoBertTokenizer

TOKENIZER_CLASSES = {
    "kobert": KoBertTokenizer,
    "distilkobert": KoBertTokenizer,
    "koelectra-base": ElectraTokenizer,
    "koelectra-small": ElectraTokenizer,
    "koelectra-base-v2": ElectraTokenizer,
    "koelectra-base-v3": ElectraTokenizer,
    "koelectra-small-v2": ElectraTokenizer,
    "koelectra-small-v3": ElectraTokenizer,
    "roberta-large": AutoTokenizer,
    "roberta-base": AutoTokenizer,
    "xlm-roberta": XLMRobertaTokenizer,
}

MODEL_FOR_TOKEN_CLASSIFICATION = {
    "kobert": TFBertForTokenClassification,
    "distilkobert": TFDistilBertForTokenClassification,
    "koelectra-base": TFElectraForTokenClassification,
    "koelectra-small": TFElectraForTokenClassification,
    "koelectra-base-v2": TFElectraForTokenClassification,
    "koelectra-base-v3": TFElectraForTokenClassification,
    "koelectra-small-v2": TFElectraForTokenClassification,
    "koelectra-small-v3": TFElectraForTokenClassification,
    "koelectra-small-v3-51000": TFElectraForTokenClassification,
    "roberta-large": TFAutoModel,
    "roberta-base": TFAutoModel,
    "xlm-roberta": TFXLMRobertaForTokenClassification,
}
