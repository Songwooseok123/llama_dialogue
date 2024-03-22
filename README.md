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



 ### dialogue act attribute model
  - input : 문장이 llama에 들어가 나온 embedding
  - output : "inform", "question", "directive", "commissive"  4개 classes 확률


### prompt를 제외하고, 생성된 token 부터 attribute 모델에 들어가게 만듬
- perturb_past의 input에 accumulated_hidden: accumulated_hidden = unpert_last_hidden[:, prompt_length:-1, :] # 1,0,4096 부터 하나씩 늘어남 

- curr_length = curr_length - prompt_length
- prediction = classifier(new_accumulated_hidden / (curr_length + 1 + horizon_length))
