# get_model_binary.py
# --hparmas, --model_binary는 default = None 인데 수정함
# --hparmas의 경우 경로고정
# --model_binary의 겨우 학습된 ckpt파일 경로 입력 필요
# --output_dir에 변환된 모델이 저장됨

import argparse
from train import KoBARTConditionalGeneration
from transformers.models.bart import BartForConditionalGeneration
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default="./logs/tb_logs/default/version_0/hparams.yaml", type=str)
parser.add_argument("--model_binary", default="./checkpoint/model_chp/epoch=01-val_loss=0.307.ckpt", type=str)
parser.add_argument("--output_dir", default='kobart_summary', type=str)
args = parser.parse_args()

inf = KoBARTConditionalGeneration.load_from_checkpoint(args.model_binary)

inf.model.save_pretrained(args.output_dir)
