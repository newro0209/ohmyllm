# OhMyLLM 🤖

일본어 GPT-2 모델을 위한 어휘 교체 및 임베딩 미세조정 도구

## 📝 프로젝트 개요

이 프로젝트는 사전학습된 `rinna/japanese-gpt2-xsmall` 모델을 기반으로 도메인 특화 어휘 교체와 임베딩 레이어 미세조정을 통해 높은 성능과 속도를 달성하는 언어 모델을 만드는 도구입니다.

## ✨ 주요 기능

- 🔄 **어휘 교체 (Vocabulary Replacement)**: 도메인 특화 토크나이저로 모델의 어휘를 교체
- 🎯 **임베딩 미세조정 (Embedding Fine-tuning)**: 새로운 어휘에 맞춰 임베딩 레이어 초기화 및 학습
- ⚡ **빠른 학습**: 전체 모델이 아닌 임베딩 레이어 중심 학습으로 빠른 도메인 적응
- 🤗 **Hugging Face 통합**: transformers, datasets, tokenizers 라이브러리 활용

## 🚀 빠른 시작

### 설치

**uv 사용 (권장):**
```bash
uv pip install -e .
```

**pip 사용:**
```bash
pip install -e .
```

### 사용법

1. **도메인 데이터 준비**
```python
# data/train.txt에 학습 데이터 준비
```

2. **토크나이저 학습**
```bash
python train_tokenizer.py --data_path data/train.txt --output_dir tokenizer/
```

3. **어휘 교체 및 임베딩 미세조정**
```bash
python train_model.py --config config/train_config.json
```

## 📁 프로젝트 구조

```
ohmyllm/
├── config/              # 설정 파일
├── src/                 # 소스 코드
│   ├── vocabulary.py    # 어휘 교체 모듈
│   ├── embedding.py     # 임베딩 미세조정 모듈
│   └── utils.py         # 유틸리티 함수
├── train_tokenizer.py   # 토크나이저 학습 스크립트
├── train_model.py       # 모델 학습 스크립트
└── pyproject.toml       # 프로젝트 설정 및 의존성
```

## 🔧 기술 스택

- Python 3.8+
- transformers
- datasets
- tokenizers
- torch

## 📄 라이선스

MIT License