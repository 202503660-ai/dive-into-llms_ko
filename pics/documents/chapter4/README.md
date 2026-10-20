# LLM 수학적 추론
본 프로젝트는 대규모 언어 모델(LLM)이 수학적 추론을 학습하도록 하는 코드를 구현하며, 핵심 기술로 지도 파인튜닝(Supervised Finetuning, SFT)을 사용하여 모델이 DeepSeek-R1이 생성한 응답을 학습하고 그로부터 복잡한 수학적 추론 기술(성찰, 검산 등)을 습득하도록 합니다.
본 실험은 [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) 데이터셋과 [Qwen/Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) 모델을 기반으로 독자들이 수학 작업에서의 데이터 증류(Data Distillation) 전 과정을 직접 체험해 볼 수 있도록 안내합니다.

## 실험 목적
- 수학 관련 데이터셋의 정제 및 전처리 과정 이해하기
- 대규모 모델 파인튜닝의 기본 방법 이해하기
- 대규모 모델을 사용한 추론의 기본 방법 이해하기
- 모델의 수학적 능력을 평가하는 기본 방법 이해하기

## 환경 요구 사항

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- vllm
- 최소 40GB VRAM 이상의 GPU

## 실험 문서

1. Jupyter notebook 열기:
```bash
jupyter notebook sft_math.ipynb
```

2. 노트북은 다음과 같은 주요 부분으로 구성되어 있습니다:
   - 데이터셋 다운로드 및 전처리
   - 모델 로드 및 학습
   - 모델 생성 및 평가

3. 구체적인 실행 방법과 설명은 노트북을 참고하세요.

## 출력 파일

학습 과정에서 학습 체크포인트(checkpoint) 파일이 생성되며, 생성이 완료된 후에는 생성 결과가 저장됩니다.

## 주의 사항

- 모델과 데이터는 [Hugging Face 공식 허브](https://huggingface.co/models) ([DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K), [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B))에서 로드하며, 해외망 접속이 불안정한 환경의 경우 미러 사이트(https://hf-mirror.com)를 활용할 수 있습니다.
- 원본 모델 파일, 데이터 파일 및 학습 후 체크포인트를 저장할 수 있는 충분한 디스크 공간을 확보하세요(최소 50GB 이상의 여유 공간 권장).
- 최소 40GB VRAM을 갖춘 GPU 그래픽 카드가 필요합니다.
