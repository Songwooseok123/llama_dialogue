# llama_dialogue

[나의 선행 연구(LLaMA의 일부 레이어 잠재공간 섭동을 통한 텍스트 생성 제어)](https://github.com/Songwooseok123/llama_pplm_layer) 의 후속 연구 입니다. 

### 선행 연구와의 차이점 
- 간단한 입력(예를 들어 "My dog"라는 입력)을 주고 단순 문장을 생성했던 선행 연구와 달리, 선행 연구에서 개발한 방법을 Dialogue Generation Task에 적용해본다.
- 선행 연구에서 해석한 결과가 Dialogue Generation Task에서도 의미있는 분석인지 확인한다. 
- 선행 연구에서 평가했던 Controllability, Perplexity, D-1,2,3 이외의 NLP Evaluation Metric도 사용한다.
- 단순 속성 정확도 평가 뿐만 아니라, 특정 속성을 가지게 될 확률 분포를 확인한다.

### 연구 목적
- 언어 모델의 텍스트 생성 시, 언어 모델의 잠재공간에 레이어 별로 섭동을 줌으로서, 언어 모델의 레이어 별 특징을 파악하고 trade-off를 해결한다.
  - Trade-off: 생성된 문장의 속성 제어도가 높을 수록 Perplexity가 높아진다. 

## 실험 상세

- 모델: Llama-7b-chat
- Task: 주어진 dialogue 문맥에 맞는 dialogue 생성 -> 발화의 특성을 제어 
- 특성
- 4개 classes: "inform", "question", "directive", "commissive"

### dialogue_generate.ipynb
- pre-trained 된 llama에 적당한 prompt를 줘서 다음 대화를 이어 나갈 수 있게 함
#### utils.prompt_make.py
- llama chat 모델은 본인을 ai라고 지칭하며 감정,행동을 가질 수 없다며 화자처럼 대답을 안 하는 경우가 종종 있음. prompot로 이거 해결  
 - ![image](https://github.com/Songwooseok123/llama_dialogue/assets/80091008/0c6c9f30-217d-43ea-a0f9-a27b9a2ea976)
 - accorging to .... (근거가 되는 코드 링크 달기)



 ### dialogue act attribute model
  - input : 문장이 llama에 들어가 나온 embedding
  - output : "inform", "question", "directive", "commissive"  4개 classes 확률


### prompt를 제외하고, 생성된 token 부터 attribute 모델에 들어가게 만듬
- perturb_past의 input에 accumulated_hidden: accumulated_hidden = unpert_last_hidden[:, prompt_length:-1, :] # 1,0,4096 부터 하나씩 늘어남 

- curr_length = curr_length - prompt_length
- prediction = classifier(new_accumulated_hidden / (curr_length + 1 + horizon_length))
### Bert classifier
- prompt 제외하고 생성된 token부터 input으로 받아서 act 분류하는 분류기 학습함,

#### attribute model, bert model 둘 다 label_balanced 하게 다시 학습함

#### 3/24 todo 
- 레이어구간 별 섭동 속도 비교하기
- 실험 setting하고 generate 하기 (no perturb 부터)
#### 실험 setting
- len
- perturb layer
- step size
- num_iterations
- 


### 실험 setting 
stepsize,perturb_layer,num_iterations
1+(12*14)개 
- not perturb
- 0.01_{100, 11,22,32} * num_iter{3,5,7}
- 0.02
- 0.03
- 0.04
- 0.05
- 0.06
- 0.07
- 0.08
- 0.09
- 0.1
- 0.2


### 실험결과
- 3/30
 - label 0,1,2,3 중 2번 label('directive')에 대한 분류가 잘 안됨.
  - 생성된 문장 평가할 때 , label 2인거 빼고 45개 문장에 대해서 평가해보는 중
  - 추가 실험(0.2,0.3,0.4,0.5,0.06,0.07,0.11,0.12,0.13,0.14,0.15,0.25,0.35) 중… 정확도 높이려고 2번 라벨 가진거 빼버렸고… 어느정도 경향이 보일 것으로 예상됨

- 4/4  추가 실험중 
