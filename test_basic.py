from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import UnigramTrainer
from tokenizers.models import Unigram
from typing import cast
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TensorType


def main():
    print("GPT-2 모델 테스트 시작...")

    # 모델과 토크나이저 로드
    model_name = "openai-community/gpt2"
    print(f"\n모델 로딩 중: {model_name}")

    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name)

    print("모델 로딩 완료!\n")

    # 테스트 프롬프트
    # prompt = "The future of artificial intelligence is"
    prompt = "인공지능의 미래는"
    print(f"입력 프롬프트: {prompt}")

    # 텍스트 인코딩
    input_ids = cast(
        torch.Tensor, tokenizer.encode(prompt, return_tensors=TensorType.PYTORCH)
    )

    # 텍스트 생성
    print("\n텍스트 생성 중...")
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    )

    # 결과 디코딩
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("생성된 텍스트:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)

    # 모델 정보 출력
    print(f"\n모델 파라미터 수: {model.num_parameters():,}")
    print(f"토크나이저 vocab 크기: {len(tokenizer)}")

    # TODO: 기존 backend_tokenizer

    return 0


if __name__ == "__main__":
    exit(main())
