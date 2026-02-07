"""
OhMyLLM 설치 스크립트
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ohmyllm",
    version="0.1.0",
    author="OhMyLLM Team",
    description="일본어 GPT-2 모델을 위한 어휘 교체 및 임베딩 미세조정 도구",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/newro0209/ohmyllm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ohmyllm-train=train_model:main",
            "ohmyllm-tokenizer=train_tokenizer:main",
        ],
    },
)
