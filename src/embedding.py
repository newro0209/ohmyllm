"""
임베딩 미세조정 모듈
임베딩 레이어를 중심으로 효율적인 미세조정을 수행
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup
)
from typing import Optional, Dict, List
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingFinetuner:
    """
    임베딩 레이어 중심의 미세조정을 수행하는 클래스
    """
    
    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: PreTrainedTokenizerFast,
        device: Optional[str] = None
    ):
        """
        Args:
            model: 미세조정할 모델
            tokenizer: 토크나이저
            device: 학습에 사용할 디바이스 (cuda/cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"사용 디바이스: {self.device}")
    
    def configure_trainable_parameters(
        self,
        train_embeddings: bool = True,
        train_lm_head: bool = True,
        train_transformer: bool = False,
        freeze_layers: Optional[List[int]] = None
    ):
        """
        학습 가능한 파라미터 설정
        
        Args:
            train_embeddings: 임베딩 레이어 학습 여부
            train_lm_head: LM Head 학습 여부
            train_transformer: Transformer 레이어 학습 여부
            freeze_layers: 고정할 Transformer 레이어 인덱스
        """
        logger.info("학습 파라미터 구성 중...")
        
        # 모든 파라미터 고정
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 임베딩 레이어 활성화
        if train_embeddings:
            for param in self.model.transformer.wte.parameters():
                param.requires_grad = True
            for param in self.model.transformer.wpe.parameters():
                param.requires_grad = True
            logger.info("✓ 임베딩 레이어 학습 활성화")
        
        # LM Head 활성화
        if train_lm_head:
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            logger.info("✓ LM Head 학습 활성화")
        
        # Transformer 레이어 활성화
        if train_transformer:
            for i, layer in enumerate(self.model.transformer.h):
                if freeze_layers and i in freeze_layers:
                    continue
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info("✓ Transformer 레이어 학습 활성화")
        
        # 학습 가능한 파라미터 수 계산
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"학습 가능 파라미터: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        save_steps: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        모델 학습
        
        Args:
            train_dataloader: 학습 데이터 로더
            num_epochs: 에포크 수
            learning_rate: 학습률
            warmup_steps: Warmup 스텝 수
            gradient_accumulation_steps: 그래디언트 누적 스텝
            max_grad_norm: 그래디언트 클리핑 값
            logging_steps: 로깅 주기
            save_steps: 모델 저장 주기
            output_dir: 모델 저장 디렉토리
        
        Returns:
            학습 기록 (loss, perplexity 등)
        """
        logger.info("학습 시작...")
        
        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        
        # 스케줄러 설정
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 학습 기록
        history = {
            "loss": [],
            "perplexity": [],
            "learning_rate": []
        }
        
        self.model.train()
        global_step = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # 배치를 디바이스로 이동
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm
                    )
                    
                    # 파라미터 업데이트
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                
                # 로깅
                epoch_loss += loss.item() * gradient_accumulation_steps
                
                if global_step % logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    perplexity = torch.exp(torch.tensor(avg_loss)).item()
                    current_lr = scheduler.get_last_lr()[0]
                    
                    history["loss"].append(avg_loss)
                    history["perplexity"].append(perplexity)
                    history["learning_rate"].append(current_lr)
                    
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "ppl": f"{perplexity:.4f}",
                        "lr": f"{current_lr:.2e}"
                    })
                
                # 모델 저장
                if save_steps and output_dir and global_step % save_steps == 0:
                    self.save_model(f"{output_dir}/checkpoint-{global_step}")
            
            # 에포크 종료 로깅
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            epoch_perplexity = torch.exp(torch.tensor(avg_epoch_loss)).item()
            logger.info(f"Epoch {epoch + 1} - Loss: {avg_epoch_loss:.4f}, "
                       f"Perplexity: {epoch_perplexity:.4f}")
        
        logger.info("학습 완료!")
        return history
    
    def save_model(self, output_dir: str):
        """
        모델과 토크나이저 저장
        
        Args:
            output_dir: 저장 디렉토리
        """
        logger.info(f"모델 저장 중: {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("저장 완료!")
    
    def evaluate(
        self,
        eval_dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        모델 평가
        
        Args:
            eval_dataloader: 평가 데이터 로더
        
        Returns:
            평가 지표
        """
        logger.info("평가 시작...")
        self.model.eval()
        
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                total_loss += outputs.loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity
        }
        
        logger.info(f"평가 결과 - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        return metrics
