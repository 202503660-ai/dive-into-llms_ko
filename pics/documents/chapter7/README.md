# LLM 텍스트 스테가노그래피(Steganography)

본 프로젝트는 대규모 언어 모델(LLM) 기반의 텍스트 스테가노그래피를 구현하며, GPT-2 모델을 사용하여 생성된 텍스트 내에 정보를 숨깁니다. 프로젝트에는 허프만 코딩(Huffman Coding)과 고정 길이 코딩(Fixed Length Coding, FLC)의 두 가지 인코딩 방법이 포함되어 있습니다.

## 주요 기능

- GPT-2 텍스트 생성 과정에서 스테가노그래피 구현
- 두 가지 인코딩 방법 지원:
  - 허프만 코딩(Huffman Coding)
  - 고정 길이 코딩(Fixed Length Coding, FLC)
- 비교를 위해 스테가노그래피 텍스트와 일반 텍스트를 동시에 생성

## 환경 요구 사항

- Python 3.8+
- PyTorch
- Transformers
- CUDA (선택 사항, GPU 가속용)

## 설치 방법

의존성 패키지 설치:

```bash
pip install torch # GPU를 사용하는 경우 GPU 버전의 torch를 지정하는 것에 유의하세요
pip install transformers
pip install jupyter
```

## 사용 방법

1. Jupyter notebook 열기:
```bash
jupyter notebook llm_stega.ipynb
```

2. 노트북은 세 가지 주요 부분으로 구성되어 있습니다:
   - 허프만 코딩 구현
   - 고정 길이 코딩(FLC) 구현
   - 텍스트 생성 및 스테가노그래피

3. 스테가노그래피 텍스트 생성:
   - 인코딩 방법 선택 (허프만 코딩 또는 FLC)
   - 파라미터 설정 (인코딩의 k값)
   - 사용자 정의 프롬프트로 메인 함수 실행

예제 코드:
```python
# 스테가노그래피 핸들러 초기화
k = 2
handle = Huffman(k=2**k, bits=bits)
# 또는 FLC 사용
# handle = FLC(k=k, bits=bits)

# 스테가노그래피가 적용된 텍스트 생성
prompt = "Hello, I'm"
stega_output, normal_output = generate_text_with_steganography(
    model, tokenizer, prompt, handle
)
```

## 출력 파일

프로그램은 두 개의 텍스트 파일을 생성합니다:
- `outputs-gpt2-stega.txt`: 스테가노그래피 텍스트
- `outputs-gpt2-normal.txt`: 스테가노그래피 제약 조건 없이 원래대로 정상 생성된 텍스트

## 주의 사항

- GPT-2 모델은 최초 실행 시 [공식 Hugging Face](https://huggingface.co/openai-community/gpt2)에서 자동으로 다운로드되며, 접속이 원활하지 않은 경우 미러 사이트(https://hf-mirror.com)를 통해 미리 다운로드하거나 다른 대규모 모델을 사용할 수도 있습니다.
- 충분한 디스크 공간(모델 약 500MB)을 확보하세요.
- 더 나은 성능을 위해 CUDA를 지원하는 GPU 사용을 권장합니다.
- 은닉할 정보의 길이는 생성되는 텍스트의 길이에 영향을 미칩니다.

## 작동 원리

1. 스테가노그래피(정보 은닉) 과정:
   - 비밀 정보를 이진 시퀀스로 변환
   - GPT-2 모델을 사용하여 텍스트 생성
   - 인코딩 방법을 통해 정보 비트를 생성된 텍스트에 삽입

2. 디코딩 과정:
   - 동일한 GPT-2 모델과 컨텍스트를 사용
   - 디코딩 방법을 통해 텍스트에서 숨겨진 정보 비트를 추출
   - 이진 시퀀스를 원래 정보로 다시 변환
