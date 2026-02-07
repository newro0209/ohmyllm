"""
모델 학습 메인 스크립트
어휘 교체 및 임베딩 미세조정을 수행
"""

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast
)
import logging

from src.vocabulary import VocabularyReplacer
from src.embedding import EmbeddingFinetuner
from src.utils import (
    TextDataset,
    load_config,
    save_config,
    create_data_collator,
    count_parameters,
    set_seed
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="일본어 GPT-2 모델 어휘 교체 및 임베딩 미세조정"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="학습 설정 파일 경로"
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    logger.info("=" * 60)
    logger.info("일본어 GPT-2 모델 어휘 교체 및 임베딩 미세조정")
    logger.info("=" * 60)
    
    # 시드 설정
    set_seed(config.get("seed", 42))
    
    # 기본 모델 및 토크나이저 로드
    logger.info("\n[1단계] 기본 모델 로드")
    base_model_name = config.get("base_model", "rinna/japanese-gpt2-xsmall")
    logger.info(f"모델: {base_model_name}")
    
    old_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    logger.info(f"기존 어휘 크기: {len(old_tokenizer)}")
    count_parameters(model)
    
    # 새로운 토크나이저 로드
    logger.info("\n[2단계] 도메인 토크나이저 로드")
    new_tokenizer_path = config.get("new_tokenizer_path")
    
    if new_tokenizer_path and Path(new_tokenizer_path).exists():
        logger.info(f"토크나이저 경로: {new_tokenizer_path}")
        new_tokenizer = PreTrainedTokenizerFast.from_pretrained(new_tokenizer_path)
        logger.info(f"새 어휘 크기: {len(new_tokenizer)}")
        
        # 어휘 교체
        logger.info("\n[3단계] 어휘 교체 수행")
        vocab_replacer = VocabularyReplacer(model, old_tokenizer, new_tokenizer)
        
        # 어휘 중복도 분석
        vocab_replacer.get_vocabulary_overlap()
        
        # 어휘 교체 실행
        reinit_strategy = config.get("reinit_strategy", "mean")
        model = vocab_replacer.replace_vocabulary(reinit_strategy=reinit_strategy)
        
        tokenizer = new_tokenizer
    else:
        logger.warning("새 토크나이저를 찾을 수 없습니다. 기존 토크나이저를 사용합니다.")
        tokenizer = old_tokenizer
    
    # 학습 데이터 준비
    logger.info("\n[4단계] 학습 데이터 준비")
    train_data_path = config.get("train_data_path")
    
    if not train_data_path or not Path(train_data_path).exists():
        logger.error(f"학습 데이터를 찾을 수 없습니다: {train_data_path}")
        return
    
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_data_path,
        block_size=config.get("block_size", 512)
    )
    
    # 데이터 로더 생성
    data_collator = create_data_collator(tokenizer, max_length=config.get("block_size", 512))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        collate_fn=data_collator
    )
    
    logger.info(f"학습 샘플 수: {len(train_dataset)}")
    logger.info(f"배치 크기: {config.get('batch_size', 4)}")
    
    # 미세조정 설정
    logger.info("\n[5단계] 임베딩 미세조정 설정")
    finetuner = EmbeddingFinetuner(
        model=model,
        tokenizer=tokenizer,
        device=config.get("device")
    )
    
    # 학습 파라미터 설정
    finetuner.configure_trainable_parameters(
        train_embeddings=config.get("train_embeddings", True),
        train_lm_head=config.get("train_lm_head", True),
        train_transformer=config.get("train_transformer", False)
    )
    
    # 학습 실행
    logger.info("\n[6단계] 모델 학습")
    history = finetuner.train(
        train_dataloader=train_dataloader,
        num_epochs=config.get("num_epochs", 3),
        learning_rate=config.get("learning_rate", 5e-5),
        warmup_steps=config.get("warmup_steps", 100),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps"),
        output_dir=config.get("output_dir", "./output")
    )
    
    # 최종 모델 저장
    logger.info("\n[7단계] 최종 모델 저장")
    output_dir = config.get("output_dir", "./output")
    finetuner.save_model(output_dir)
    
    # 학습 기록 저장
    history_path = Path(output_dir) / "training_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    logger.info(f"학습 기록 저장: {history_path}")
    
    # 설정 저장
    config_path = Path(output_dir) / "train_config.json"
    save_config(config, str(config_path))
    
    logger.info("\n" + "=" * 60)
    logger.info("학습 완료!")
    logger.info("=" * 60)
    logger.info(f"모델 저장 위치: {output_dir}")
    
    # 테스트 생성
    logger.info("\n[테스트] 텍스트 생성")
    test_prompt = config.get("test_prompt", "こんにちは")
    logger.info(f"프롬프트: {test_prompt}")
    
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(finetuner.device)
        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"생성된 텍스트: {generated_text}")


if __name__ == "__main__":
    main()
