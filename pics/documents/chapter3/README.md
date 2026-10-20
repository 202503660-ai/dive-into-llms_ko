# 밑바닥부터 시작하는 대규모 모델: 대규모 모델 지식 편집
들어가며: 언어 모델의 편집 방법과 도구
> 특정 지식에 대한 언어 모델의 기억을 제어하고 싶으신가요? 적절한 편집 방법을 선택하여 특정 지식을 편집하고, 편집된 모델을 검증해 봅시다!

## 1. 본 튜토리얼의 목표:

- EasyEdit 툴킷 사용법 익히기
- 언어 모델 편집 방법(기초) 마스터하기
- 다양한 유형의 편집 방법 선택 기준 및 응용 시나리오 이해하기

## 2. 사전 준비:
### 2.1 EasyEdit 알아보기

https://github.com/zjunlp/EasyEdit

EasyEdit는 GPT-J, Llama, GPT-NEO, GPT2, T5 등 언어 모델을 편집하기 위한 Python 패키지로, 다른 입력에 대한 성능에 부정적인 영향을 주지 않으면서 특정 지식에 대해 언어 모델의 행동을 효과적으로 변경하는 것을 목표로 하며, 사용하기 쉽고 확장하기 편리합니다.

EasyEdit는 현재 널리 사용되는 편집 방법들을 통합하여 제공합니다:
![](./assets/1.png)

### 2.2 주요 프레임워크

![](./assets/2.png)
EasyEdit는 통합된 Editor, Method 및 Evaluate 프레임워크를 포함하며, 각각 편집 시나리오, 편집 기술 및 평가 방법을 나타냅니다.
- Editor: 작업 시나리오를 기술하며, 편집할 모델, 편집할 지식 및 기타 필요한 하이퍼파라미터를 포함합니다.
- Method: 사용하는 구체적인 지식 편집 방법(예: ROME, MEND 등)입니다.
- Evaluate: 지식 편집 성능을 평가하는 지표로, 신뢰성(Reliability), 일반화(Generalization), 지역성(Locality), 이식성(Portability)을 포함합니다.
- Trainer: 일부 편집 방법은 일정한 학습 과정을 필요로 하며, 이는 Trainer 모듈을 통해 구현됩니다.
## 3. 환경 설치:
```
git clone https://github.com/zjunlp/EasyEdit.git
(선택 사항) conda create -n EasyEdit python=3.9.7
cd EasyEdit
pip install -r requirements.txt
```
## 4. 편집 실습 사례
> 목표: GPT-2-XL의 지식 기억을 변경하여 리오넬 메시(Lionel Messi)의 종목을 축구에서 농구로 변경하기 (football->basketball). 
단계:
- 편집 방법 선택 및 파라미터 준비
- 지식 편집을 위한 데이터 준비
- Editor 인스턴스화
- 실행(Run)!
아래에서는 ROME 방법을 예시로 구체적으로 소개합니다:
### 4.1 ROME
Jupyter Notebook: [https://colab.research.google.com/drive/1KkyWqyV3BjXCWfdrrgbR-QS3AAokVZbr?usp=sharing#scrollTo=zWfGkNb9FBJQ] 
- 편집 방법 선택 및 파라미터 준비
  - 편집 방법으로 ROME을 선택하고, ROME과 GPT2-XL에 필요한 파라미터를 준비합니다.
  - 예: alg_name: "ROME", model_name: "./hugging_cache/gpt2-xl" 또는 로컬 모델 경로, "device": 사용할 GPU 번호
  - 나머지 파라미터는 기본값으로 유지할 수 있습니다.
![](./assets/3.png)
- 지식 편집을 위한 데이터 준비
    ```
    prompts = ['Question:What sport does Lionel Messi play? Answer:'] # x_e
    ground_truth = ['football'] # y
    target_new = ['basketball'] # y_e
    subject = ['Lionel Messi'] 
    ```
- Editor 인스턴스화: 준비된 파라미터를 BaseEditor 클래스에 전달하여 인스턴스화하고, 커스텀 Editor 인스턴스를 얻습니다.
    ```
    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2-xl.yaml')
    editor=BaseEditor.from_hparams(hparams)
    ```
- 실행(Run)! editor의 edit 메서드를 호출합니다:
    ```
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )
    ```
![](./assets/4.png)
> 참고: 특정 모델을 처음 편집할 때는 Wiki 코퍼스를 다운로드하고 해당 모델의 각 레이어 상태를 계산하여 저장(stats_dir: "./data/stats")한 뒤, 이후 편집에서 재사용합니다. 따라서 첫 편집 시에는 시간이 다소 소요될 수 있으니 네트워크가 원활한 상태에서 잠시 기다려 주세요.
### 4.2 검증 및 평가
editor.edit는 metrics(EasyEdit의 Evaluate 모듈에서 계산됨)를 반환합니다. 형식은 다음과 같습니다:
![](./assets/5.png)
일반화(Generalization), 지역성(Locality), 이식성(Portability) 수치를 얻으려면 edit 메서드에 평가용 데이터를 전달해야 합니다.

지역성(Locality)을 예로 들면, 아래 코드를 통해 edit 메서드가 지역성 지표(즉, locality_inputs에 대한 모델 답변의 정확도)를 계산하게 됩니다.
```
locality_inputs = {
    'neighborhood':{
        'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
        'ground_truth': ['piano', 'basketball', 'Finnish']
    }
}
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    locality_inputs=locality_inputs,
    keep_original_weight=False
)
```
또는 편집 전후 모델의 생성(generate) 동작을 직접 비교할 수도 있습니다.
```
generation_prompts = [
    "Lionel Messi, the",
    "The law in Ikaalinen declares the language"
]

model = GPT2LMHeadModel.from_pretrained('./hugging_cache/gpt2').to('cuda')
batch = tokenizer(generation_prompts, return_tensors='pt', padding=True, max_length=30)

pre_edit_outputs = model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
    max_new_tokens=3
)
post_edit_outputs = edited_model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
    max_new_tokens=3)
```
## 5. 대규모 편집 (선택 사항)
### 5.1 배치 편집 (Batch edit)
여러 개의 데이터를 병렬 리스트 형태로 구성하여 edit 메서드에 동시에 전달함으로써 배치 편집(batch edit)을 수행할 수 있으며, 이 경우 MEMIT가 가장 적합한 방법입니다. (https://colab.research.google.com/drive/1P1lVklP8bTyh8uxxSuHnHwB91i-1LW6Z)
```
prompts = ['Question:What sport does Lionel Messi play? Answer:',
            'The law in Ikaalinen declares the language']
ground_truth = ['football', 'Finnish']
target_new = ['basketball', 'Swedish']
subject = ['Lionel Messi', 'Ikaalinen']
```
### 5.2 벤치마크(Benchmark) 테스트
- Counterfact
- ZsRE
```
{
    "case_id": 4402,
    "pararel_idx": 11185,
    "requested_rewrite": {
      "prompt": "{} debuted on",
      "relation_id": "P449",
      "target_new": {
        "str": "CBS",
        "id": "Q43380"
      },
      "target_true": {
        "str": "MTV",
        "id": "Q43359"
      },
      "subject": "Singled Out"
    },
    "paraphrase_prompts": [
      "No one on the ground was injured.  v",
      "The sex ratio was 1063. Singled Out is to debut on"
    ],
    "neighborhood_prompts": [
      "Daria premieres on",
      "Teen Wolf was originally aired on",
      "Spider-Man: The New Animated Series was originally aired on",
      "Celebrity Deathmatch premiered on",
      "\u00c6on Flux premiered on",
      "My Super Psycho Sweet 16 premieres on",
      "Daria was released on",
      "Jersey Shore premiered on",
      "Skins was originally aired on",
      "All You've Got premiered on"
    ]
  }
  ```
https://github.com/zjunlp/EasyEdit/blob/main/examples/run_zsre_llama2.py 
