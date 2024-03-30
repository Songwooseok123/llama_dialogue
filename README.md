# llama_dialogue

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
