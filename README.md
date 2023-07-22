# Finetuning of Falcon-7B LLM using QLoRA on Mental Health Dataset

## Introduction:
Mental health issues are often misunderstood or not fully grasped by the general public. This lack of understanding can lead to fear, discomfort, and negative perceptions about mental health conditions. Media portrayals of mental health often perpetuate negative stereotypes, leading to misconceptions and fear. Overcoming mental health stigma requires a multi-faceted approach that involves education, raising awareness, promoting empathy and understanding, challenging stereotypes, and ensuring accessible and quality mental health care.
Mental health directly impacts an individual's overall well-being, quality of life, and ability to function effectively in daily life. Good mental health is essential for experiencing happiness, fulfilment, and a sense of purpose. Mental health and physical health are closely intertwined. Untreated mental health issues can lead to or worsen physical health problems, such as cardiovascular diseases, weakened immune systems, and chronic conditions.

## Core Rationale:
Chatbots offer a readily available and accessible platform for individuals seeking support. They can be accessed anytime and anywhere, providing immediate assistance to those in need. Chatbots can offer empathetic and non-judgmental responses, providing emotional support to users. While they cannot replace human interaction entirely, they can be a helpful supplement, especially in moments of distress.<br><br>
NOTE: _It is important to note that while mental health chatbots can be helpful, they are not a replacement for professional mental health care. They can complement existing mental health services by providing additional support and resources._

## Dataset:
Dataset was curated from online FAQs related to mental health, popular healthcare blogs like WebMD and Healthline, and other wiki articles related to mental health. The dataset was pre-processed in a conversational format such that both questions asked by the patient and responses given by the doctor are in the same text. The dataset for this mental health conversational AI can be found here: [heliosbrahma/mental_health_conversational_dataset](https://huggingface.co/datasets/heliosbrahma/mental_health_conversational_dataset).<br><br>
NOTE: _All questions and answers have been anonymized to remove any PII data and preprocessed to remove any unwanted characters._

## Model Finetuning:
This is the major step in the entire project. I have used sharded Falcon-7B pre-trained model and finetuned it to using the QLoRA technique on my custom mental health dataset. The entire finetuning process took less than an hour and it was finetuned entirely on free GPU T4 from Google Colab.
The rationale behind using sharded base model is mentioned in my blog post: [Fine-tuning of Falcon-7B Large Language Model using QLoRA on Mental Health Dataset](https://medium.com/@iamarunbrahma/fine-tuning-of-falcon-7b-large-language-model-using-qlora-on-mental-health-dataset-aa290eb6ec85)<br>
Adding here the training loss metrics tracking report from WandB monitoring logs for 180 steps training run: [train/loss forQLoRA fine-tuning](https://api.wandb.ai/links/heliosbrahma/j3c2w00t) <br><br>
NOTE: _Try changing hyperparameters in TrainingArguments and LoraConfig based on your requirements. With the settings mentioned in notebook, I achieved 0.42 training loss after 180 steps._

## Model Inference:
PEFT fine-tuned model has been updated here: [heliosbrahma/falcon-7b-finetuned-mental-health-conversational](https://huggingface.co/heliosbrahma/falcon-7b-finetuned-mental-health-conversational). You can directly load the model using the following settings:<br>
```python
query = "<MENTION YOUR QUERY>"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftConfig, PeftModel

PEFT_MODEL = "heliosbrahma/falcon-7b-finetuned-mental-health-conversational"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
config = PeftConfig.from_pretrained(PEFT_MODEL)
peft_base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
peft_model = PeftModel.from_pretrained(peft_base_model, PEFT_MODEL)

peft_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
peft_tokenizer.pad_token = peft_tokenizer.eos_token

peft_encoding = peft_tokenizer(query, return_tensors="pt").to(device)
peft_outputs = peft_model.generate(input_ids=peft_encoding.input_ids, generation_config=GenerationConfig(max_new_tokens=200, pad_token_id = peft_tokenizer.eos_token_id, eos_token_id = peft_tokenizer.eos_token_id, attention_mask = peft_encoding.attention_mask, temperature=0.7, top_p=0.7, num_return_sequences=1,))
peft_text_output = peft_tokenizer.decode(peft_outputs[0], skip_special_tokens=True)
print(peft_text_output)
```
It takes less than 3 minutes to generate the model response. The response will be more structured and give a better response than the response generated from the base model.

## Conclusion:
I have written a detailed technical blog explaining key concepts of QLoRA and PEFT fine-tuning method: [Fine-tuning of Falcon-7B Large Language Model using QLoRA on Mental Health Dataset](https://medium.com/@iamarunbrahma/fine-tuning-of-falcon-7b-large-language-model-using-qlora-on-mental-health-dataset-aa290eb6ec85). If you still have any queries, you can open an issue on this repo or comment on my blog.
