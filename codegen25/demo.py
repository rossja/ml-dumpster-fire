from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '../../models/Salesforce_codegen25-7b-multi'

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)

text = "def hello_world():"
input_ids = tokenizer(text, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids, max_length=128)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
