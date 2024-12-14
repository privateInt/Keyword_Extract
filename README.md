# 프로젝트 목적

- keyword extract 모델 fine-tuning
- 이 project는 관세청 RAG 챗봇의 성능을 보충하기 위한 방법으로 선정된 지식그래프에 keyword가 사용되기 때문에 진행
- 적은 GPU로 기능을 수행하기 위해, LLM이 아닌 KoBART로 진행
- 의도한 모델의 입력과 출력 예시: 이사화물 통관예약을 하면 뭐가 좋나요?(입력) -> 이사화물, 통관, 예약, 장점(출력)

# Reference

- [KoBART-summarization](https://github.com/seujung/KoBART-summarization)

# Data

- KorQuad 데이터 셋, 관세청 PoC 사업 데이터 셋의 질문에 chatGPT로 label을 만든 데이터 75,403쌍

# 기능

- KoBART를 fine-tuning 할 수 있다. huggingface에 있는 KoBART 계열 모델이면 학습 가능하다.(default: gogamza/kobart-base-v1)
- user input(질문)입력시 user input에서 추출한 keyword를 response로 return하는 서버(flask)를 띄울 수 있다.
- user input(질문)을 입력하고 response를 화면에 출력하는 데모 페이지(streamlit)를 띄울 수 있다.

# 파일 구조
```sh
project
├── data
│   ├── train.tsv
│   └── test.tsv
├── dataset.py
├── demo.py
├── get_model_binary.py
├── inf_server.py
├── model.py
├── requirements.txt
└── train.py
```

# 폴더 및 파일 역할
| 폴더 및 파일 | 설명 |
|------|--------|
|data|fine-tuning 데이터 파일을 저장하는 폴더, 현재 tsv파일을 읽고 처리하는 방식이며, tsv파일의 column에는 "query", "keyword"가 포함돼야 한다. train 데이터는 train.tsv, test 데이터는 test.tsv파일을 읽으며 파일 이름을 바꾸고 싶은 경우 train.py에서 변경 가능|
|dataset.py|"query", "keyword" column을 가지는 tsv파일을 입력 받아 pytorch lightning data module로 반환하는 Dataset을 정의한 파일|
|model.py|KoBART model을 정의한 파일|
|train.py|데이터를 사용해 KoBART model을 fine-tuning한다. wandb API KEY 입력시 loss 추적 가능|
|get_model_binary.py|pytorch-lightning binary --> huggingface binary로 추출|
|inf_server.py.py|KoBART의 fine-tuning 결과를 이용해 inference server(flask)를 작동|
|demo.py|inference server에 입력값을 전송하고 return값을 출력하는 데모 페이지(streamlit)작동|


# 환경
- GPU: A100(40GiB)GPU * 4
- 저장 모델 용량: 1GB
- python 3.8
- CUDA Version 12.0
- Nvidia Driver Version 525.105.17
  
![Cap 2024-05-30 09-31-10-647](https://github.com/privateInt/RAG-chatbot/assets/95892797/72d2fe19-8af6-4dc0-993e-9b6c958173d7)

# fine-tuning
- hyper parameter
  
| hyper parameter | value |
|------|--------|
|epochs|2|
|batch_size|8|
|learning_rate|3e-5|
|max_len|512|

![Cap 2024-04-18 17-18-21-964](https://github.com/user-attachments/assets/71532de2-30ba-4357-b75a-783d7b59c795)


# 명령어

<table border="1">
  <tr>
    <th>내용</th>
    <th>명령어</th>
  </tr>
  <tr>
    <td>환경 설치</td>
    <td>pip install -r requirements.txt</td>
  </tr>
  <tr>
    <td>fine-tuning</td>
    <td>python train.py --gradient_clip_val 1.0 \
                --max_epochs 2 \
                --checkpoint checkpoint \
                --accelerator gpu \
                --num_gpus 4 \
                --batch_size 8 \
                --num_workers 4</td>
  </tr>
  <tr>
    <td>pytorch-lightning binary --> huggingface binary로 추출</td>
    <td>python get_model_binary.py --hparams ./logs/tb_logs/default/version_0/hparams.yaml --model_binary ./checkpoint/model_chp/epoch=01-val_loss=0.307.ckpt</td>
  </tr>
  <tr>
    <td>fine-tuning 결과를 이용해 inference server(flask)를 작동</td>
    <td>python inf_server.py</td>
  </tr>
  <tr>
    <td>inference server에 입력값을 전송하고 return값을 출력하는 데모 페이지(streamlit)작동</td>
    <td>streamlit run demo.py</td>
  </tr>
  
</table>

# 데모 페이지 예시

![Cap 2024-04-19 16-52-10-944](https://github.com/user-attachments/assets/28ea04a7-78d6-4c26-a55d-32f9028fa0f8)
- 다소 장난스러워 보일 수 있으나 학습하지 않은 데이터에 대해 주어, 동사, 목적어를 명사형으로 추출할 수 있는지 demo page에서 테스트

# 결론
- fine-tuning을 진행한 base model은 summarization model로, 키워드를 추출하는 것 또한 요약이라는 가설을 증명할 수 있었음.
- inference server의 GPU memory usage는 4GB로, 의도했던 것처럼 적은 GPU 자원으로 구동이 가능했음.
- model의 입출력이 의도한대로 나오는 것을 확인했으며, test.tsv 파일에 대한 inference 진행후 정성평가 결과 95%데이터가 사용가능 했음.
- 지식그래프를 이용한 Retrieval의 성능은 top1 acc 20%에 그쳤다. 이음동의어가 원인이었으며, 이를 해결하기 위해선 domain 단어 간의 관계 정립이 필요할 것으로 추측된다.
