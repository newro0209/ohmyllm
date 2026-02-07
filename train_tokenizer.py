"""
토크나이저 학습 스크립트
도메인 데이터로 새로운 토크나이저를 학습
"""

import argparse
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_tokenizer(
    data_path: str,
    output_dir: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    special_tokens: list = None
):
    """
    BPE 토크나이저 학습

    Args:
        data_path: 학습 데이터 경로
        output_dir: 출력 디렉토리
        vocab_size: 어휘 크기
        min_frequency: 최소 빈도수
        special_tokens: 특수 토큰 리스트
    """
    logger.info("=" * 50)
    logger.info("토크나이저 학습 시작")
    logger.info("=" * 50)

    # 특수 토큰 설정
    if special_tokens is None:
        special_tokens = [
            "<pad>",
            "<s>",
            "</s>",
            "<unk>",
            "<mask>"
        ]

    logger.info(f"학습 데이터: {data_path}")
    logger.info(f"어휘 크기: {vocab_size}")
    logger.info(f"최소 빈도수: {min_frequency}")
    logger.info(f"특수 토큰: {special_tokens}")

    # 토크나이저 초기화
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Pre-tokenizer 설정 (공백 기반)
    tokenizer.pre_tokenizer = Whitespace()

    # Trainer 설정
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )

    # 학습 실행
    logger.info("토크나이저 학습 중...")
    files = [data_path]
    tokenizer.train(files, trainer)

    # Post-processor 설정
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 토크나이저 저장
    tokenizer_path = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    logger.info(f"토크나이저 저장: {tokenizer_path}")

    # Hugging Face 형식으로 변환 및 저장
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>"
    )

    fast_tokenizer.save_pretrained(str(output_path))
    logger.info(f"Hugging Face 토크나이저 저장: {output_path}")

    # 테스트
    test_text = "이것은 테스트 문장입니다."
    tokens = fast_tokenizer.tokenize(test_text)
    ids = fast_tokenizer.encode(test_text)

    logger.info(f"\n테스트 문장: {test_text}")
    logger.info(f"토큰화 결과: {tokens}")
    logger.info(f"인코딩 결과: {ids}")

    logger.info("\n" + "=" * 50)
    logger.info("토크나이저 학습 완료!")
    logger.info("=" * 50)

    return fast_tokenizer


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="도메인 특화 토크나이저 학습"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="학습 데이터 파일 경로"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tokenizer",
        help="토크나이저 저장 디렉토리"
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="어휘 크기"
    )

    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="토큰 최소 빈도수"
    )

    args = parser.parse_args()

    # 토크나이저 학습
    train_tokenizer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency
    )


if __name__ == "__main__":
    main()
