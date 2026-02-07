"""
유틸리티 함수 모듈
데이터 로딩, 전처리 등 공통 기능 제공
"""

import json
import torch
from pathlib import Path
from typing import Optional, Dict, List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    텍스트 데이터셋 클래스
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        file_path: str,
        block_size: int = 512,
        overwrite_cache: bool = False
    ):
        """
        Args:
            tokenizer: 토크나이저
            file_path: 텍스트 파일 경로
            block_size: 최대 시퀀스 길이
            overwrite_cache: 캐시 덮어쓰기 여부
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # 캐시 파일 경로
        cache_file = Path(file_path).with_suffix('.cache')
        
        if cache_file.exists() and not overwrite_cache:
            logger.info(f"캐시 파일 로딩: {cache_file}")
            self.examples = torch.load(cache_file)
        else:
            logger.info(f"데이터 파일 처리 중: {file_path}")
            
            # 텍스트 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 토크나이징
            tokenized = tokenizer.encode(text)
            
            # 블록 단위로 분할
            self.examples = []
            for i in range(0, len(tokenized) - block_size + 1, block_size):
                self.examples.append(tokenized[i:i + block_size])
            
            # 캐시 저장
            logger.info(f"캐시 저장 중: {cache_file}")
            torch.save(self.examples, cache_file)
        
        logger.info(f"데이터셋 크기: {len(self.examples)} 샘플")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.examples[idx], dtype=torch.long),
            'attention_mask': torch.ones(len(self.examples[idx]), dtype=torch.long)
        }


def load_config(config_path: str) -> Dict:
    """
    설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
    
    Returns:
        설정 딕셔너리
    """
    logger.info(f"설정 파일 로딩: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def save_config(config: Dict, output_path: str):
    """
    설정 파일 저장
    
    Args:
        config: 설정 딕셔너리
        output_path: 저장 경로
    """
    logger.info(f"설정 파일 저장: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def create_data_collator(tokenizer: PreTrainedTokenizerFast, max_length: int = 512):
    """
    데이터 콜레이터 생성
    
    Args:
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이
    
    Returns:
        데이터 콜레이터 함수
    """
    def collate_fn(examples):
        # 입력 시퀀스 패딩
        input_ids = [ex['input_ids'] for ex in examples]
        attention_mask = [ex['attention_mask'] for ex in examples]
        
        # 배치로 변환
        max_len = min(max([len(ids) for ids in input_ids]), max_length)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            if len(ids) > max_len:
                ids = ids[:max_len]
                mask = mask[:max_len]
            else:
                padding_length = max_len - len(ids)
                ids = torch.cat([
                    ids,
                    torch.full((padding_length,), tokenizer.pad_token_id, dtype=torch.long)
                ])
                mask = torch.cat([
                    mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ])
            
            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask)
        }
    
    return collate_fn


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    모델의 파라미터 수 계산
    
    Args:
        model: PyTorch 모델
    
    Returns:
        파라미터 통계
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    stats = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params
    }
    
    logger.info(f"총 파라미터: {total_params:,}")
    logger.info(f"학습 가능 파라미터: {trainable_params:,}")
    logger.info(f"고정 파라미터: {stats['frozen_params']:,}")
    
    return stats


def load_text_file(file_path: str, encoding: str = 'utf-8') -> List[str]:
    """
    텍스트 파일 로드
    
    Args:
        file_path: 파일 경로
        encoding: 인코딩
    
    Returns:
        텍스트 라인 리스트
    """
    logger.info(f"텍스트 파일 로딩: {file_path}")
    with open(file_path, 'r', encoding=encoding) as f:
        lines = [line.strip() for line in f if line.strip()]
    logger.info(f"로드된 라인 수: {len(lines)}")
    return lines


def prepare_training_corpus(
    file_paths: List[str],
    output_path: str,
    shuffle: bool = True
):
    """
    학습 코퍼스 준비
    
    Args:
        file_paths: 입력 파일 경로 리스트
        output_path: 출력 파일 경로
        shuffle: 섞기 여부
    """
    logger.info("학습 코퍼스 준비 중...")
    
    all_lines = []
    for file_path in file_paths:
        lines = load_text_file(file_path)
        all_lines.extend(lines)
    
    if shuffle:
        import random
        random.shuffle(all_lines)
    
    logger.info(f"총 라인 수: {len(all_lines)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line + '\n')
    
    logger.info(f"코퍼스 저장 완료: {output_path}")


def set_seed(seed: int = 42):
    """
    랜덤 시드 설정
    
    Args:
        seed: 시드 값
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"랜덤 시드 설정: {seed}")
