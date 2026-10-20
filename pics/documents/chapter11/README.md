# 실습으로 배우는 대규모 모델: RLHF

> 본 실습 매뉴얼은 온라인 자료 [blog](https://newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html) 및 [trl examples](https://github.com/huggingface/trl/blob/v0.8.0/examples/notebooks/gpt2-sentiment.ipynb)를 번역 및 통합한 것입니다.

재현 실험 환경 구성: 단일 카드 NVIDIA A800-SXM4-80GB에서 10097MiB 점유, 학습 소요 시간 35분 19초.

튜토리얼 읽기: [slide](./RLHF.pdf)

notebook: [notebook](./RLHF.ipynb)

## PPO 작동 방식
1. Rollout: 언어 모델이 query에 따라 응답을 생성합니다.
2. Evaluation: 쿼리와 응답은 함수, 모델, 인간 피드백 또는 이들의 조합을 사용하여 평가됩니다. 이 과정은 각 쿼리/응답 쌍에 대해 하나의 **스칼라 값**을 생성해야 합니다.
3. Optimization: 최적화 단계에서는 쿼리/응답 쌍을 사용하여 시퀀스 내 토큰의 로그 확률을 계산합니다. 이는 학습 중인 모델과 참조 모델(Reference Model)을 통해 수행됩니다. 두 출력 간의 KL 발산(KL-divergence)은 생성된 응답이 참조 언어 모델에서 너무 멀리 벗어나지 않도록 보장하는 추가 보상 신호로 사용됩니다. 그런 다음 PPO를 사용하여 활성 언어 모델을 학습합니다.
<div style="text-align: center">
<img src='figs/trl1.png' width='600'>
<p style="text-align: center;"> <b>그림:</b> PPO 흐름도 </p>
</div>

# 긍정적인 리뷰를 생성하도록 GPT-2 파인튜닝하기
> BERT 감성 분류기를 보상 함수로 사용하여 긍정적인 IMDB 영화 리뷰를 생성하도록 GPT-2를 최적화합니다.

<div style="text-align: center">
<img src='figs/gpt2_bert_training.png' width='600'>
<p style="text-align: center;"> <b>그림:</b> GPT-2 파인튜닝 실험 설정</p>
</div>

IMDB 데이터셋을 기반으로 긍정적인 영화 리뷰를 생성하도록 GPT-2를 파인튜닝합니다. 이 모델은 실제 리뷰의 앞부분을 입력받아 긍정적인 후속 내용을 생성해야 합니다. 긍정적인 후속 내용에 보상을 주기 위해 BERT 분류기를 사용하여 생성된 문장의 감성을 분석하고, 분류기의 출력을 PPO 학습의 보상 신호로 사용합니다.

## 실험 설정

### 모델 및 데이터 다운로드
> 기본적으로 공식 Hugging Face 허브에서 바로 다운로드할 수 있으며, 해외망 연결이 제한된 환경의 경우 `export HF_ENDPOINT=https://hf-mirror.com` 설정을 함께 사용할 수 있습니다.

데이터셋
```bash
huggingface-cli download --resume-download stanfordnlp/imdb --local-dir dataset/imdb --repo-type dataset
```
참조 모델
```bash
huggingface-cli download --resume-download lvwerra/gpt2-imdb --local-dir model/gpt2-imdb
```
보상 모델
```bash
huggingface-cli download --resume-download lvwerra/distilbert-imdb --local-dir model/distilbert-imdb
```

### 의존성 임포트


```python
# %pip install -r requirements.txt
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
```


```python
import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
```

### 설정


```python
config = PPOConfig(
    model_name="model/gpt2-imdb",
    learning_rate=1.41e-5,
    log_with="wandb",
)

sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
```


```python
import wandb

wandb.init()
```

여기서 `gpt2_imdb`라는 이름의 GPT-2 모델을 로드한 것을 볼 수 있습니다. 이 모델은 Hugging Face의 [스크립트](https://github.com/huggingface/transformers/blob/v4.40.0/examples/legacy/run_language_modeling.py)(별도 특수 설정 없음)를 사용하여 IMDB 데이터셋에서 1 에폭 동안 추가 파인튜닝된 모델입니다. 나머지 파라미터는 주로 원 논문인 <[Fine-Tuning Language Models from Human Preferences](https://huggingface.co/papers/1909.08593)>에서 가져왔습니다. 이 모델과 BERT 모델은 모두 Hugging Face 모델 허브에서 가져올 수 있으며 구체적인 링크는 [여기](https://huggingface.co/models)에 있습니다.

## 데이터 및 모델 로드

### IMDB 데이터셋 로드
IMDB 데이터셋은 50,000개의 영화 리뷰를 포함하며 "긍정"/"부정" 감성 피드백이 라벨링되어 있습니다. IMDB 데이터셋을 DataFrame으로 로드하고 최소 200자 이상인 리뷰만 필터링합니다. 그런 다음 각 텍스트를 토크나이징하고 `LengthSampler`를 사용하여 지정된 길이로 무작위 절단(truncate)합니다.


```python
def build_dataset(
    config,
    dataset_name="dataset/imdb",
    input_min_text_length=2,
    input_max_text_length=8,
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds
```


```python
dataset = build_dataset(config)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
```

### 사전학습된 GPT2 언어 모델 로드
가치 헤드(value head)가 있는 GPT2 모델과 토크나이저를 로드합니다. 모델을 두 번 로드하는데, 첫 번째 모델은 최적화에 사용되고 두 번째 모델은 시작점과의 KL 발산(KL-divergence)을 계산하기 위한 참조 모델(reference model)로 사용됩니다. 이는 최적화된 모델이 원래 언어 모델에서 너무 멀리 벗어나지 않도록 보장하기 위해 PPO 학습에서 추가 보상 신호로 사용됩니다.


```python
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token
```

### PPOTrainer 초기화
`PPOTrainer`는 이후의 디바이스 할당 및 최적화를 담당합니다:


```python
ppo_trainer = PPOTrainer(
    config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator
)
```


### BERT 분류기 로드
IMDB 데이터셋에서 파인튜닝된 BERT 분류기를 로드합니다.


```python
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
sentiment_pipe = pipeline(
    "sentiment-analysis", model="model/distilbert-imdb", device=device
)
```

    Device set to use cuda:0


모델이 출력하는 것은 부정 클래스와 긍정 클래스의 로짓(logits)입니다. 긍정 클래스의 로짓을 언어 모델의 보상 신호로 사용합니다.


```python
text = "this movie was really bad!!"
sentiment_pipe(text, **sent_kwargs)
```




    [{'label': 'NEGATIVE', 'score': 2.3350484371185303},
     {'label': 'POSITIVE', 'score': -2.726576089859009}]




```python
text = "this movie was really good!!"
sentiment_pipe(text, **sent_kwargs)
```




    [{'label': 'POSITIVE', 'score': 2.557040214538574},
     {'label': 'NEGATIVE', 'score': -2.294790267944336}]



### 생성 설정
응답 생성을 위해 샘플링 방법만 사용하며, top-k와 뉴클리어스 샘플링(nucleus sampling)을 끄고 최소 길이를 설정합니다.


```python
gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
```

## 모델 최적화

### 학습 루프

학습 루프는 다음과 같은 주요 단계로 구성됩니다:
1. 정책 네트워크(GPT-2)에서 쿼리 응답 가져오기
2. BERT에서 쿼리/응답의 감성 점수 가져오기
3. (쿼리, 응답, 보상) 삼중항을 활용하여 PPO로 정책 최적화하기


```python
output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)


generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}


for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    #### Get response from gpt2
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        query_response = ppo_trainer.generate(query, **generation_kwargs).squeeze()
        response_len = len(query_response) - len(query)
        response_tensors.append(query_response[-response_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    positive_scores = [
        item["score"]
        for output in pipe_outputs
        for item in output
        if item["label"] == "POSITIVE"
    ]
    rewards = [torch.tensor(score) for score in positive_scores]

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
```

      0%|          | 0/194 [00:00<?, ?it/s]The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
      4%|▍         | 8/194 [01:23<32:18, 10.42s/it]You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
    100%|██████████| 194/194 [35:19<00:00, 10.92s/it]


### 학습 진행 상황
Weights & Biases를 사용하여 학습 진행 상황을 추적하고 있다면 아래 그림과 유사한 곡선을 볼 수 있을 것입니다. wandb.ai의 인터랙티브 예제 보고서를 확인해 보세요: [링크](https://wandb.ai/huggingface/trl/runs/w9l3110g).  
<div style="text-align: center">
<img src='figs/gpt2_tuning_progress.png' width='800'>
<p style="text-align: center;"> <b>그림:</b> 학습 중 평균 보상의 변화 추이 </p>
</div>  
몇 번의 최적화 단계를 거친 후 모델이 더 긍정적인 출력을 생성하기 시작하는 것을 관찰할 수 있습니다.  

## 모델 검사
IMDB 데이터셋에서 몇 가지 예제를 확인해 봅시다. `ref_model`을 사용하여 최적화된 모델 `model`과 최적화 이전 모델을 비교할 수 있습니다.


```python
#### get a batch from the dataset
bs = 16
game_data = dict()
dataset.set_format("pandas")
df_batch = dataset[:].sample(bs)
game_data["query"] = df_batch["query"].tolist()
query_tensors = df_batch["input_ids"].tolist()

response_tensors_ref, response_tensors = [], []

#### get response from gpt2 and gpt2_ref
for i in range(bs):
    query = torch.tensor(query_tensors[i]).to(device)

    gen_len = output_length_sampler()
    query_response = ref_model.generate(
        query.unsqueeze(0), max_new_tokens=gen_len, **gen_kwargs
    ).squeeze()
    response_len = len(query_response) - len(query)
    response_tensors_ref.append(query_response[-response_len:])

    query_response = model.generate(
        query.unsqueeze(0), max_new_tokens=gen_len, **gen_kwargs
    ).squeeze()
    response_len = len(query_response) - len(query)
    response_tensors.append(query_response[-response_len:])

#### decode responses
game_data["response (before)"] = [
    tokenizer.decode(response_tensors_ref[i]) for i in range(bs)
]
game_data["response (after)"] = [
    tokenizer.decode(response_tensors[i]) for i in range(bs)
]

#### sentiment analysis of query/response pairs before/after
texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
positive_scores = [
    item["score"]
    for output in pipe_outputs
    for item in output
    if item["label"] == "POSITIVE"
]
game_data["rewards (before)"] = positive_scores

texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
positive_scores = [
    item["score"]
    for output in pipe_outputs
    for item in output
    if item["label"] == "POSITIVE"
]
game_data["rewards (after)"] = positive_scores

# store results in a dataframe
df_results = pd.DataFrame(game_data)
df_results
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>query</th>
      <th>response (before)</th>
      <th>response (after)</th>
      <th>rewards (before)</th>
      <th>rewards (after)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Well I guess I know</td>
      <td>that Cantor may be an</td>
      <td>..but I loved it</td>
      <td>0.230196</td>
      <td>2.281557</td>
    </tr>
    <tr>
      <th>1</th>
      <td>This is an excellent,</td>
      <td>direct-to-video film with typical</td>
      <td>enjoyable movie.&lt;|endoftext|&gt;</td>
      <td>2.846593</td>
      <td>2.840860</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Now, I</td>
      <td>'ve never had the chance with James</td>
      <td>loved the growing episode - and the</td>
      <td>0.656194</td>
      <td>2.525894</td>
    </tr>
    <tr>
      <th>3</th>
      <td>We tend</td>
      <td>not to see Arthur</td>
      <td>to like this very</td>
      <td>-0.280880</td>
      <td>2.183822</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The proverb "Never judge a book</td>
      <td>by the cover" has caught on. After glancing t...</td>
      <td>with high compliments, but it is recommended ...</td>
      <td>0.274649</td>
      <td>2.065951</td>
    </tr>
    <tr>
      <th>5</th>
      <td>I've never understood</td>
      <td>why so many artsmen,</td>
      <td>this film but it's delightful</td>
      <td>0.835574</td>
      <td>2.782384</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hugh (Ed Harris) is</td>
      <td>an acclaimed "hero" and his fian</td>
      <td>a wonderful actor who is a good adaptation</td>
      <td>1.580167</td>
      <td>2.602940</td>
    </tr>
    <tr>
      <th>7</th>
      <td>This particular Joe McDoakes</td>
      <td>' episode brought all the wrong bits and</td>
      <td>movie is really a great movie. It</td>
      <td>0.870956</td>
      <td>2.795245</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sisters In</td>
      <td>Vrooms 8.23, I signed up for all of the</td>
      <td>The Universe 1: Sunny is cute, and has a cute...</td>
      <td>1.175259</td>
      <td>2.062330</td>
    </tr>
    <tr>
      <th>9</th>
      <td>I was very fond of this</td>
      <td>film, it was obviously a bad idea when first ...</td>
      <td>show, and know that I have seen it several times</td>
      <td>1.058164</td>
      <td>2.511273</td>
    </tr>
    <tr>
      <th>10</th>
      <td>If he wanted to be</td>
      <td>funny, he could</td>
      <td>a genius eventually,</td>
      <td>-0.388943</td>
      <td>0.405888</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Thats My</td>
      <td>Grade...&lt;br /&gt;&lt;br /&gt;Although</td>
      <td>Way was the best movie that I watched.</td>
      <td>-0.151680</td>
      <td>2.473050</td>
    </tr>
    <tr>
      <th>12</th>
      <td>This is possibly the best short</td>
      <td>film I have come across in almost two years.</td>
      <td>film ever written. It has some very memorable...</td>
      <td>2.511835</td>
      <td>2.775994</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Some people say this is</td>
      <td>exactly what happens in Hollywood; where come...</td>
      <td>a powerful film to listen to. It really captures</td>
      <td>0.637631</td>
      <td>2.821085</td>
    </tr>
    <tr>
      <th>14</th>
      <td>A remake of</td>
      <td>"The Wizard of Oz</td>
      <td>the legendary Kingan oil</td>
      <td>0.292409</td>
      <td>0.434021</td>
    </tr>
    <tr>
      <th>15</th>
      <td>What a terrible</td>
      <td>movie!&lt;|endoftext|&gt;</td>
      <td>chopping sounded so good, I love it! We have a</td>
      <td>-2.681461</td>
      <td>2.340650</td>
    </tr>
  </tbody>
</table>


생성된 시퀀스의 보상 평균/중앙값을 관찰하면 뚜렷한 차이를 확인할 수 있습니다.


```python
print("mean:")
display(df_results[["rewards (before)", "rewards (after)"]].mean())
print()
print("median:")
display(df_results[["rewards (before)", "rewards (after)"]].median())
```

    mean:

    rewards (before)    0.591666
    rewards (after)     2.243934
    dtype: float64


​    
    median:

    rewards (before)    0.646912
    rewards (after)     2.492161
    dtype: float64


## 모델 저장
마지막으로 추후 사용을 위해 모델을 저장합니다.


```python
model.save_pretrained("model/gpt2-imdb-pos-v2")
tokenizer.save_pretrained("model/gpt2-imdb-pos-v2")
```




    ('model/gpt2-imdb-pos-v2/tokenizer_config.json',
     'model/gpt2-imdb-pos-v2/special_tokens_map.json',
     'model/gpt2-imdb-pos-v2/vocab.json',
     'model/gpt2-imdb-pos-v2/merges.txt',
     'model/gpt2-imdb-pos-v2/added_tokens.json',
     'model/gpt2-imdb-pos-v2/tokenizer.json')
