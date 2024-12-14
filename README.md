# 프로젝트 목적

- keyword extract 모델 fine-tuning
- 이 project는 관세청 RAG 챗봇의 성능을 보충하기 위한 방법으로 선정된 지식그래프에 keyword가 사용되기 때문에 진행
- 적은 GPU로 기능을 수행하기 위해, LLM이 아닌 KoBART로 진행

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



# 한계 및 극복방안

- 데이터 부분: 이 project에서는 사람이 직접 데이터를 제작했다. 추후 일반화하기 위해서는 데이터 추출 자동화가 선행돼야 한다.
- Retrieval 성능: top1의 acc는 30%에 그쳤다. Retrieval 성능을 올리기 위해 BM25같은 sparse retriever와 ensemble retriever 구축이 필요하다. 또한 보조 모델을 활용하는 등 topK가 커져도 gold passage를 추출하는 기능이 필요하다.
- LLM 성능: Retrieval이 잘못된 정보를 전달할 경우 LLM이 필터링할 수 있어야 한다.

# 앞으로의 방향

![Cap 2024-02-13 16-59-36-937](https://github.com/privateInt/RAG-chatbot/assets/95892797/d6161384-2ccf-4b8b-b644-cf5fe1da77de)

![Cap 2024-02-13 17-00-56-248](https://github.com/privateInt/RAG-chatbot/assets/95892797/1778fc2f-26f0-42b1-a878-fb4e0a2a5fe9)

## 데이터 pipeline
- 문서에는 자연어와 표, 이미지 등이 섞여있음
- 표의 경우 MD, HTML 등의 형태로 변경하는 등 LLM 학습 가능한 형태로 1차 가공 필요
- 1차 가공이 끝난 데이터를 출처,제시문,질문,답변 등으로 구분하는 2차 가공 필요

## advanced RAG
- Multi-Query: 질문 표현 방식에 따라 retrieval 성능이 바뀌므로 같은 의미지만 다른 표현을 여러개 생성하여 retriever 필요
- Self-Query: 조건식으로 검색 대상을 최대한 좁힌 후 retriever 시행
- Time-Weighted: meta data(문서 생성 시기, 문서 종류 등)를 활용해 검색 대상을 최대한 좁힌 후 retriever 시행
- Ensemble-Retriever: sparse retriever의 성능이 더 좋은 경우가 존재하기 때문에 구축 필요
- Long Context Reorder: 연관성이 높은 문서를 일부러 맨앞, 맨뒤에 Reorder

![Cap 2024-02-14 12-21-48-066](https://github.com/privateInt/RAG-chatbot/assets/95892797/3214e493-fb28-4af1-a190-8f7e9dac049d)

## graph DB
- advanced RAG만으로 Retriever 성능이 부족할 수 있음, 추가 retriever system 구축이 필요함
- graph DB는 노드(키워드)와 엣지(관계)로 이루어짐
- 관계 정의 가이드라인 필요
- 키워드 추출 가이드라인 필요

## chain of thought
- prompt를 단계별로 제공하여 LLM이 문제를 쉽게 이해할 수 있도록 조치 필요

## 더 살펴보고 싶은 논문 및 내용
- Self-RAG
- Re-ranker
- DPO trainer
- RLHF (RM, PPO)
