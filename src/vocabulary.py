"""
어휘 교체 모듈
도메인 특화 토크나이저로 기존 모델의 어휘를 교체하는 기능 제공
"""

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VocabularyReplacer:
    """
    사전학습된 모델의 어휘를 새로운 토크나이저의 어휘로 교체하는 클래스
    """

    def __init__(
        self,
        model: GPT2LMHeadModel,
        old_tokenizer: PreTrainedTokenizerFast,
        new_tokenizer: PreTrainedTokenizerFast
    ):
        """
        Args:
            model: 어휘를 교체할 모델
            old_tokenizer: 기존 토크나이저
            new_tokenizer: 새로운 토크나이저
        """
        self.model = model
        self.old_tokenizer = old_tokenizer
        self.new_tokenizer = new_tokenizer

    def replace_vocabulary(
        self,
        reinit_strategy: str = "random",
        token_mapping: Optional[Dict[str, str]] = None
    ) -> GPT2LMHeadModel:
        """
        모델의 어휘를 교체하고 임베딩을 초기화

        Args:
            reinit_strategy: 임베딩 초기화 전략 ("random", "mean", "map")
            token_mapping: 토큰 매핑 정보 (reinit_strategy가 "map"일 때 사용)

        Returns:
            어휘가 교체된 모델
        """
        logger.info("어휘 교체 시작...")

        old_vocab_size = len(self.old_tokenizer)
        new_vocab_size = len(self.new_tokenizer)

        logger.info(f"기존 어휘 크기: {old_vocab_size}")
        logger.info(f"새로운 어휘 크기: {new_vocab_size}")

        # 모델의 임베딩 크기 조정
        self.model.resize_token_embeddings(new_vocab_size)

        # 임베딩 초기화
        if reinit_strategy == "random":
            self._reinit_random()
        elif reinit_strategy == "mean":
            self._reinit_mean()
        elif reinit_strategy == "map" and token_mapping:
            self._reinit_map(token_mapping)
        else:
            raise ValueError(f"지원하지 않는 초기화 전략: {reinit_strategy}")

        logger.info("어휘 교체 완료!")
        return self.model

    def _reinit_random(self):
        """랜덤 초기화 전략"""
        logger.info("랜덤 초기화 전략 적용")
        # 기본적으로 resize_token_embeddings가 랜덤 초기화를 수행
        pass

    def _reinit_mean(self):
        """평균 초기화 전략 - 기존 임베딩의 평균으로 새 토큰 초기화"""
        logger.info("평균 초기화 전략 적용")

        with torch.no_grad():
            # 입력 임베딩
            input_embeddings = self.model.get_input_embeddings()
            old_vocab_size = len(self.old_tokenizer)
            new_vocab_size = len(self.new_tokenizer)

            if new_vocab_size > old_vocab_size:
                # 기존 임베딩의 평균 계산
                mean_embedding = input_embeddings.weight[:old_vocab_size].mean(dim=0)

                # 새로운 토큰에 평균 임베딩 할당
                input_embeddings.weight[old_vocab_size:] = mean_embedding

            # 출력 임베딩도 동일하게 처리
            output_embeddings = self.model.get_output_embeddings()
            if new_vocab_size > old_vocab_size:
                mean_embedding = output_embeddings.weight[:old_vocab_size].mean(dim=0)
                output_embeddings.weight[old_vocab_size:] = mean_embedding

    def _reinit_map(self, token_mapping: Dict[str, str]):
        """
        매핑 초기화 전략 - 토큰 매핑 정보를 사용하여 초기화

        Args:
            token_mapping: 새 토큰 -> 기존 토큰 매핑
        """
        logger.info("매핑 초기화 전략 적용")

        with torch.no_grad():
            input_embeddings = self.model.get_input_embeddings()
            output_embeddings = self.model.get_output_embeddings()

            for new_token, old_token in token_mapping.items():
                new_id = self.new_tokenizer.convert_tokens_to_ids(new_token)
                old_id = self.old_tokenizer.convert_tokens_to_ids(old_token)

                if new_id is not None and old_id is not None:
                    # 기존 토큰의 임베딩을 새 토큰에 복사
                    input_embeddings.weight[new_id] = input_embeddings.weight[old_id]
                    output_embeddings.weight[new_id] = output_embeddings.weight[old_id]

    def get_vocabulary_overlap(self) -> Dict[str, any]:
        """
        기존 어휘와 새 어휘의 중복도 분석

        Returns:
            중복도 통계 정보
        """
        old_vocab = set(self.old_tokenizer.get_vocab().keys())
        new_vocab = set(self.new_tokenizer.get_vocab().keys())

        overlap = old_vocab & new_vocab
        only_old = old_vocab - new_vocab
        only_new = new_vocab - old_vocab

        stats = {
            "old_vocab_size": len(old_vocab),
            "new_vocab_size": len(new_vocab),
            "overlap_size": len(overlap),
            "overlap_ratio": len(overlap) / len(old_vocab) if old_vocab else 0,
            "only_old_size": len(only_old),
            "only_new_size": len(only_new)
        }

        logger.info(f"어휘 중복도: {stats['overlap_ratio']:.2%}")
        logger.info(f"공통 토큰: {stats['overlap_size']}")
        logger.info(f"기존 어휘에만 존재: {stats['only_old_size']}")
        logger.info(f"새 어휘에만 존재: {stats['only_new_size']}")

        return stats
