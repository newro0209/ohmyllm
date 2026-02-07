# 사용법 가이드

## 목차
1. [설치](#설치)
2. [빠른 시작](#빠른-시작)
3. [상세 설명](#상세-설명)
4. [고급 사용법](#고급-사용법)

## 설치

### 필수 요구사항
- Python 3.8 이상
- PyTorch 2.0 이상
- CUDA (GPU 사용 시)

### 패키지 설치
```bash
# 저장소 클론
git clone https://github.com/newro0209/ohmyllm.git
cd ohmyllm

# 의존성 설치
pip install -r requirements.txt

# 또는 패키지 설치
pip install -e .
```

## 빠른 시작

### 1. 학습 데이터 준비
`data/train.txt` 파일에 도메인 특화 텍스트 데이터를 준비합니다.

```
こんにちは、世界！
人工知能は急速に発展しています。
...
```

### 2. 도메인 토크나이저 학습 (선택사항)
기존 토크나이저를 사용할 수도 있지만, 도메인 특화 성능을 위해 새로운 토크나이저를 학습할 수 있습니다.

```bash
python train_tokenizer.py \
    --data_path data/train.txt \
    --output_dir tokenizer \
    --vocab_size 32000 \
    --min_frequency 2
```

### 3. 모델 학습
```bash
python train_model.py --config config/train_config.json
```

## 상세 설명

### 어휘 교체 (Vocabulary Replacement)

어휘 교체는 사전학습된 모델의 토크나이저를 도메인 특화 토크나이저로 교체하는 과정입니다.

**초기화 전략:**
- `random`: 랜덤 초기화 (기본값)
- `mean`: 기존 임베딩의 평균으로 초기화
- `map`: 토큰 매핑을 사용한 초기화

설정 파일에서 `reinit_strategy`를 변경하여 전략을 선택할 수 있습니다.

### 임베딩 미세조정 (Embedding Fine-tuning)

임베딩 레이어를 중심으로 효율적인 미세조정을 수행합니다.

**학습 가능한 파라미터 설정:**
```json
{
  "train_embeddings": true,    // 임베딩 레이어
  "train_lm_head": true,        // 언어 모델 헤드
  "train_transformer": false    // Transformer 레이어
}
```

### 설정 파일 (config/train_config.json)

```json
{
  "base_model": "rinna/japanese-gpt2-xsmall",  // 기본 모델
  "new_tokenizer_path": "./tokenizer",          // 새 토크나이저 경로
  "train_data_path": "./data/train.txt",        // 학습 데이터 경로
  "output_dir": "./output",                     // 출력 디렉토리
  
  "reinit_strategy": "mean",    // 임베딩 초기화 전략
  "block_size": 512,            // 시퀀스 최대 길이
  "batch_size": 4,              // 배치 크기
  
  "num_epochs": 3,              // 에포크 수
  "learning_rate": 5e-5,        // 학습률
  "warmup_steps": 100,          // Warmup 스텝
  "gradient_accumulation_steps": 1,  // 그래디언트 누적
  "max_grad_norm": 1.0,         // 그래디언트 클리핑
  
  "logging_steps": 10,          // 로깅 주기
  "save_steps": 500             // 저장 주기
}
```

## 고급 사용법

### Python API 사용

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from src import VocabularyReplacer, EmbeddingFinetuner

# 모델 및 토크나이저 로드
old_tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-xsmall")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-xsmall")

# 새 토크나이저 로드
from transformers import PreTrainedTokenizerFast
new_tokenizer = PreTrainedTokenizerFast.from_pretrained("./tokenizer")

# 어휘 교체
replacer = VocabularyReplacer(model, old_tokenizer, new_tokenizer)
stats = replacer.get_vocabulary_overlap()
model = replacer.replace_vocabulary(reinit_strategy="mean")

# 미세조정
finetuner = EmbeddingFinetuner(model, new_tokenizer)
finetuner.configure_trainable_parameters(
    train_embeddings=True,
    train_lm_head=True
)

# 학습
history = finetuner.train(train_dataloader, num_epochs=3)

# 저장
finetuner.save_model("./output")
```

### 커스텀 데이터셋 사용

```python
from src.utils import TextDataset
from torch.utils.data import DataLoader

# 데이터셋 생성
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="custom_data.txt",
    block_size=512
)

# 데이터 로더
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### 모델 평가

```python
# 평가 데이터 로더 준비
eval_dataloader = DataLoader(eval_dataset, batch_size=4)

# 평가 실행
metrics = finetuner.evaluate(eval_dataloader)
print(f"Loss: {metrics['eval_loss']:.4f}")
print(f"Perplexity: {metrics['eval_perplexity']:.4f}")
```

### 텍스트 생성

```python
import torch

model.eval()
with torch.no_grad():
    # 입력 텍스트
    prompt = "こんにちは"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # 생성
    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    
    # 디코딩
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
```

## 성능 최적화 팁

### GPU 메모리 절약
- `batch_size`를 줄입니다
- `gradient_accumulation_steps`를 증가시킵니다
- `block_size`를 줄입니다

### 학습 속도 향상
- Mixed precision training 사용:
  ```python
  from torch.cuda.amp import autocast, GradScaler
  ```
- 더 큰 `batch_size` 사용
- 데이터 로더에서 `num_workers` 증가

### 모델 성능 향상
- 더 많은 학습 데이터 준비
- `num_epochs` 증가
- 학습률 튜닝
- Transformer 레이어도 함께 학습 (`train_transformer=true`)

## 문제 해결

### CUDA out of memory
배치 크기를 줄이거나 그래디언트 누적을 사용하세요:
```json
{
  "batch_size": 2,
  "gradient_accumulation_steps": 2
}
```

### 학습이 느림
- GPU를 사용하고 있는지 확인
- 데이터 로더 최적화
- Mixed precision training 사용

### 생성 품질이 낮음
- 더 많은 에포크 학습
- 도메인 데이터 품질 확인
- 다른 초기화 전략 시도

## 참고 자료
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Rinna 일본어 GPT-2](https://huggingface.co/rinna/japanese-gpt2-xsmall)
- [Tokenizers 라이브러리](https://huggingface.co/docs/tokenizers)
