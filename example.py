"""
간단한 사용 예제
기본 설정으로 모델을 학습하고 텍스트를 생성하는 예제
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import set_seed


def main():
    """메인 함수"""
    print("=" * 60)
    print("OhMyLLM 간단한 사용 예제")
    print("=" * 60)

    # 시드 설정
    set_seed(42)

    # 1. 기본 모델 로드
    print("\n[1] 기본 모델 로드...")
    model_name = "rinna/japanese-gpt2-xsmall"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"✓ 모델 로드 완료: {model_name}")
    print(f"✓ 어휘 크기: {len(tokenizer)}")

    # 2. 텍스트 생성 (미세조정 전)
    print("\n[2] 미세조정 전 텍스트 생성...")
    prompt = "こんにちは"
    print(f"프롬프트: {prompt}")

    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"생성 결과: {generated_text}")

    print("\n" + "=" * 60)
    print("예제 완료!")
    print("=" * 60)
    print("\n다음 단계:")
    print("1. 도메인 데이터를 data/train.txt에 준비")
    print("2. python train_tokenizer.py로 토크나이저 학습")
    print("3. python train_model.py --config config/train_config.json로 모델 학습")


if __name__ == "__main__":
    main()
