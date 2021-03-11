# from torch.optim.lr_scheduler import SAVE
import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer



# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)



# print(tokenizer.decode(tokenizer.eos_token_id)) 

sentence="Chris Evans"
input_ids=tokenizer.encode(sentence,return_tensors='pt')

# print(input_ids)
# print(tokenizer.decode(input_ids[0][1]))

output=model.generate(input_ids,max_length=500,num_beams=5,no_repeat_ngram_size=2,early_stopping=True)

# print(tokenizer.decode(output[0],skip_special_tokens=True))
# print(output)

text=tokenizer.decode(output[0],skip_special_tokens=True)

with open('blogpost.txt','w') as f:
    f.write(text)
print(text)