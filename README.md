# OhMyLLM

GPT-2 모델 어휘 교체 및 임베딩 미세조정 도구

## 설치

```bash
uv pip install -e .
```

## 사용법

### 1. 토크나이저 학습
```bash
python train_tokenizer.py --data_path data/corpus.txt --output_dir tokenizer/
```

### 2. 모델 학습
```bash
python train_model.py --config config/train_config.json
```

### 3. Python에서 사용
```python
from src.vocabulary import replace_vocabulary
from src.embedding import finetune_embeddings

# 어휘 교체
model, tokenizer = replace_vocabulary(
    base_model="openai-community/gpt2",
    new_tokenizer_path="./tokenizer",
    reinit_strategy="mean"
)

# 임베딩 미세조정
finetune_embeddings(
    model=model,
    tokenizer=tokenizer,
    train_data_path="./data/corpus.txt",
    output_dir="./output",
    num_epochs=3,
    batch_size=4
)
```

## 설정 (config/train_config.json)

```json
{
  "base_model": "openai-community/gpt2",
  "new_tokenizer_path": "./tokenizer",
  "train_data_path": "./data/corpus.txt",
  "output_dir": "./output",
  "reinit_strategy": "mean",
  "block_size": 512,
  "batch_size": 4,
  "num_epochs": 3,
  "learning_rate": 5e-5,
  "train_embeddings": true,
  "train_lm_head": true,
  "train_transformer": false
}
```

## 프로젝트 구조

```
├── config/              # 설정 파일
├── src/                 # 소스 코드
│   ├── vocabulary.py    # 어휘 교체
│   ├── embedding.py     # 임베딩 미세조정
│   └── utils.py         # 유틸리티
├── train_tokenizer.py   # 토크나이저 학습
└── train_model.py       # 모델 학습
```
