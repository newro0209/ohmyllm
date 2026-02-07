"""
단위 테스트
코드의 기본 기능을 검증하는 테스트
"""

import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """모듈 임포트 테스트"""
    print("테스트 1: 모듈 임포트...")
    try:
        from src import (  # noqa: F401
            VocabularyReplacer,
            EmbeddingFinetuner,
            TextDataset,
            load_config,
            save_config,
            set_seed
        )
        print("✓ 모든 모듈 임포트 성공")
        return True
    except Exception as e:
        print(f"✗ 임포트 실패: {e}")
        return False


def test_config_functions():
    """설정 파일 함수 테스트"""
    print("\n테스트 2: 설정 파일 함수...")
    try:
        from src.utils import load_config, save_config
        import tempfile
        import json

        # 테스트 설정
        test_config = {
            "test_key": "test_value",
            "test_number": 42
        }

        # 임시 파일에 저장
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
            json.dump(test_config, f)

        # 로드 테스트
        loaded_config = load_config(temp_file)

        # 검증
        assert loaded_config == test_config, "설정 로드 실패"

        # 저장 테스트
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file2 = f.name

        save_config(test_config, temp_file2)

        # 다시 로드하여 검증
        reloaded_config = load_config(temp_file2)
        assert reloaded_config == test_config, "설정 저장/로드 실패"

        # 임시 파일 삭제
        os.unlink(temp_file)
        os.unlink(temp_file2)

        print("✓ 설정 파일 함수 테스트 성공")
        return True
    except Exception as e:
        print(f"✗ 테스트 실패: {e}")
        return False


def test_seed_setting():
    """시드 설정 테스트"""
    print("\n테스트 3: 랜덤 시드 설정...")
    try:
        from src.utils import set_seed
        import torch
        import random

        # 시드 설정
        set_seed(42)

        # 랜덤 값 생성
        random_val1 = random.random()
        torch_val1 = torch.rand(1).item()

        # 시드 다시 설정
        set_seed(42)

        # 같은 랜덤 값이 나와야 함
        random_val2 = random.random()
        torch_val2 = torch.rand(1).item()

        assert random_val1 == random_val2, "Random 시드 설정 실패"
        assert torch_val1 == torch_val2, "Torch 시드 설정 실패"

        print("✓ 랜덤 시드 설정 테스트 성공")
        return True
    except Exception as e:
        print(f"✗ 테스트 실패: {e}")
        return False


def test_parameter_counting():
    """파라미터 카운팅 테스트"""
    print("\n테스트 4: 파라미터 카운팅...")
    try:
        from src.utils import count_parameters
        import torch.nn as nn

        # 간단한 모델 생성
        model = nn.Sequential(
            nn.Linear(10, 20),  # 10*20 + 20 = 220 parameters
            nn.Linear(20, 5)    # 20*5 + 5 = 105 parameters
        )
        # 총 325 parameters

        stats = count_parameters(model)

        assert stats['total_params'] == 325, "파라미터 수 계산 오류"
        assert stats['trainable_params'] == 325, "학습 가능 파라미터 수 오류"
        assert stats['frozen_params'] == 0, "고정 파라미터 수 오류"

        # 일부 파라미터 고정
        for param in model[0].parameters():
            param.requires_grad = False

        stats = count_parameters(model)

        assert stats['trainable_params'] == 105, "고정 후 학습 가능 파라미터 수 오류"
        assert stats['frozen_params'] == 220, "고정 파라미터 수 오류"

        print("✓ 파라미터 카운팅 테스트 성공")
        return True
    except Exception as e:
        print(f"✗ 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("OhMyLLM 단위 테스트")
    print("=" * 60)

    results = []

    # 테스트 실행
    results.append(("모듈 임포트", test_imports()))
    results.append(("설정 파일 함수", test_config_functions()))
    results.append(("랜덤 시드 설정", test_seed_setting()))
    results.append(("파라미터 카운팅", test_parameter_counting()))

    # 결과 출력
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n총 {total}개 테스트 중 {passed}개 통과")

    if passed == total:
        print("\n모든 테스트 통과! ✓")
        return 0
    else:
        print(f"\n{total - passed}개 테스트 실패")
        return 1


if __name__ == "__main__":
    exit(main())
