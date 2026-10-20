# 실습으로 배우는 대규모 모델: GUI 에이전트 구축
## 본 튜토리얼 목표
1. GUI 에이전트 분야의 기술 노선과 연구 현황 파악하기

2. 오픈소스 모델 Qwen2-VL-7B를 기반으로 OS-Kairos 데이터셋을 활용하여 적응형 인간-컴퓨터 상호작용(HCI) GUI 에이전트를 구축하는 방법 실습하기

3. 구축한 GUI 에이전트를 기반으로 추론 수행하기
## 1. 사전 준비
### 1.1 GUI 에이전트 분야의 기술 노선과 연구 현황 알아보기
튜토리얼 읽기: [[강의자료 (Slides)](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter9/GUIagent.pdf)]

### 1.2 적응형 인간-컴퓨터 상호작용 GUI 에이전트란 무엇인가
참고 논문: [[Paper](https://arxiv.org/abs/2503.16465)]

### 1.3 데이터셋 준비
본 튜토리얼에서는 OS-Kairos를 예시로 삼아, 이 연구에서 오픈소스로 공개한 데이터셋을 기반으로 간단한 GUI 에이전트를 구축하고 추론을 수행합니다.

[[Data](https://github.com/Wuzheng02/OS-Kairos)]의 README.md 파일에 있는 다운로드 링크에서 데이터셋을 다운로드하고 환경에 압축을 해제합니다.

데이터 형식 예시는 다음과 같습니다:
```json
    {
        "task": "넷이즈 클라우드 뮤직(NetEase Cloud Music)을 열고 <Shape of You>를 검색한 뒤 이 노래를 재생하세요.",
        "image_path": "/data1/wuzh/cloud_music/images/1736614680.6518524_1.png",
        "list": [
            " 넷이즈 클라우드 뮤직 열기  ",
            " 홈 화면 상단의 검색창 클릭  ",
            " 입력: Shape of You  ",
            " 올바른 검색 결과 선택  ",
            " 곡 제목을 클릭하여 재생  "
        ],
        "now_step": 1,
        "previous_actions": [
            "CLICK <point>[[381,367]]</point>"
        ],
        "score": 5,
        "osatlas_action": "CLICK <point>[[454,87]]</point>",
        "teacher_action": "CLICK <point>[[500,100]]</point>",
        "success": false
    },
```
### 1.4 모델 준비
[[Model](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)]에서 Qwen2-VL-7B 모델 가중치를 다운로드합니다.

## 1.5 지도 파인튜닝(SFT) 코드 준비
본 튜토리얼에서는 LLaMA-Factory를 사용하여 Qwen2-VL-7B에 대한 지도 파인튜닝을 수행하므로, 먼저 [[Code](https://github.com/hiyouga/LLaMA-Factory/)]에서 소스 코드를 다운로드합니다.

## 2. GUI 에이전트 구축
### 2.1 데이터 전처리
다음 코드를 `get_sharpgpt.py`로 저장하고, `Kairos_train.json` 경로와 전처리된 학습 세트 json을 저장할 경로를 입력한 뒤 파일을 실행합니다.
```python
import json


with open('', 'r', encoding='utf-8') as f:
    data = json.load(f)


preprocessed_data = []
for item in data:
    task = item.get("task", "")
    previous_actions = item.get("previous_actions", "") 
    image_path = item.get("image_path","")
    prompt_text = f"""
    You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs and give a score. Your skill set includes both basic and custom actions:

    1. Basic Actions
    Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
    Basic Action 1: CLICK 
        - purpose: Click at the specified position.
        - format: CLICK <point>[[x-axis, y-axis]]</point>
        - example usage: CLICK <point>[[101, 872]]</point>

    Basic Action 2: TYPE
        - purpose: Enter specified text at the designated location.
        - format: TYPE [input text]
        - example usage: TYPE [Shanghai shopping mall]

    Basic Action 3: SCROLL
        - Purpose: SCROLL in the specified direction.
        - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
        - Example Usage: SCROLL [UP]

    2. Custom Actions
    Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

    Custom Action 1: PRESS_BACK
        - purpose: Press a back button to navigate to the previous screen.
        - format: PRESS_BACK
        - example usage: PRESS_BACK

    Custom Action 2: PRESS_HOME
        - purpose: Press a home button to navigate to the home page.
        - format: PRESS_HOME
        - example usage: PRESS_HOME

    Custom Action 3: ENTER
        - purpose: Press the enter button.
        - format: ENTER
        - example usage: ENTER

    Custom Action 4: IMPOSSIBLE
        - purpose: Indicate the task is impossible.
        - format: IMPOSSIBLE
        - example usage: IMPOSSIBLE

    In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action.

    And I hope you evaluate your action to be scored, giving it a score from 1 to 5. 
    A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.
    1 means you believe this action definitely cannot achieve the goal.
    2 means you believe this action is very unlikely to achieve the goal.
    3 means you believe this action has a certain chance of achieving the goal.
    4 means you believe this action is very likely to achieve the goal.
    5 means you believe this action will definitely achieve the goal.

    And your final goal, previous actions, and associated screenshot are as follows:
    Final goal: {task}
    previous actions: {previous_actions}
    screenshot:<image>
    Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks.(where osatlas action must be one of the action formats I provided and score must be 1 to 5):
    action:
    score:
    """

    action = item.get("osatlas_action", "")
    score = item.get("score", "")
    preprocessed_item = {
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            },
            {
                "role": "assistant",
                "content": f"action: {action}\nscore:{score}",
            }
        ],
        "images":[image_path]
    }
    preprocessed_data.append(preprocessed_item)


with open('', 'w', encoding='utf-8') as f:
    json.dump(preprocessed_data, f, ensure_ascii=False, indent=4)
```
이 단계를 통해 OS-Kairos의 학습 세트를 sharegpt 형식에 맞는 데이터로 성공적으로 변환하여 다음 학습 단계를 원활히 진행할 수 있도록 데이터 전처리를 완료했습니다.
## 2.2 지도 파인튜닝(SFT)
이전 단계에서 LLaMA-Factory 학습 형식에 맞춘 Kairos 데이터셋을 얻었습니다. 이제 데이터셋을 등록하고 학습 정보를 설정하기 위해 LLaMA-Factory를 수정해야 합니다.
먼저 `data/dataset_info.json`을 수정하여 다음 내용을 추가해 데이터셋을 등록합니다:
```json
"Karios" :{
  "file_name": "Karios_qwenscore.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages",
    "images": "images"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
},   
```
이어서 `/examples/train_full/qwen2vl_full_sft.yaml`을 수정하여 학습 정보를 설정합니다:
```yaml
### model
model_name_or_path: 
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: Karios
template: qwen2_vl
cutoff_len: 4096
max_samples: 9999999
#max_samples: 999
overwrite_cache: true
preprocessing_num_workers: 4

### output
output_dir: 
logging_steps: 10
save_steps: 60000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 20000
```
위는 예시 설정이며, `model_name_or_path`는 Qwen2-VL-7B의 경로이고 `output_dir`는 체크포인트와 로그를 저장할 경로입니다.
설정이 완료되면 학습을 실행할 수 있으며, 예시 명령줄은 다음과 같습니다:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2vl_full_sft.yaml
```
여기서는 최소 3장의 80GB A100 컴퓨팅 리소스를 사용해야 합니다.

## 3. OS-Kairos 추론 검증
학습이 완료되면 추론을 진행할 수 있습니다. 추론 또한 다양한 방법을 선택할 수 있는데, `transformers` 라이브러리와 `torch` 라이브러리를 통해 직접 추론 코드를 작성할 수도 있고, LLaMA-Factory를 사용하여 원클릭 추론을 수행할 수도 있습니다. 아래에서는 LLaMA-Factory를 사용하여 원클릭 추론을 수행하는 방법을 시연합니다.
먼저 `/examples/inference/qwen2_vl.yaml`을 수정해야 하며, 예시는 다음과 같습니다:
```yaml
model_name_or_path: 
template: qwen2_vl
```
여기서 `model_name_or_path`는 학습 후 저장된 체크포인트 경로입니다.
그런 다음 아래 명령어를 통해
```
CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli webchat examples/inference/qwen2_vl.yaml
```
원클릭 추론을 진행할 수 있습니다. OS-Kairos 테스트 세트의 이미지나 개인 스마트폰의 스크린샷을 준비하여 `get_sharpgpt.py` 형식의 텍스트 프롬프트와 함께 에이전트에 입력하면, OS-Kairos는 추론 후 현재 수행해야 한다고 판단하는 액션(action)과 해당 액션에 대한 신뢰도 점수(score)를 제시합니다. 신뢰도 점수가 낮을수록 현재 명령이 OS-Kairos의 능력 범위를 벗어났음을 의미하며, 인간-컴퓨터 상호작용을 위해 인간의 개입이 필요함을 나타냅니다.