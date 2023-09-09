import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = '/models/transformers/Salesforce/xgen-7b-8k-inst'

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16)

header = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
)

article = """
Rightly speaking, it is improper, not to say illegal, for those sadly privileged few who go in and out of Broadmoor Criminal Lunatic Asylum, to have pointed out to them any particular character, however notorious he may have been or to what heights of public interest his infamy had carried him, before the testifying doctors and a merciful jury consigned him to this place without hope. But often had John Flack been pointed out as he shuffled about the grounds, his hands behind him, his chin on his breast, a tall, lean old man in an ill-fitting suit of drab clothing, who spoke to nobody and was spoken to by few.
“That is Flack⁠—The Flack⁠—the cleverest crook in the world.⁠ ⁠… Crazy John Flack⁠ ⁠… nine murders⁠ ⁠…”
In their queer, sane moments, men who were in Broadmoor for isolated homicides were rather proud of Old John. The officers who locked him up at night and watched him as he slept had little to say against him, because he gave no trouble, and through all the six years of his incarceration had never once been seized of those frenzies which so often end in the hospital for some poor innocent devil, and a rubber-padded cell for the frantic author of misfortune.
"""
prompt = f"### Human: Please summarize the following article.\n\n{article}.\n###"

inputs = tokenizer(header + prompt, return_tensors="pt")
sample = model.generate(**inputs, do_sample=True, max_new_tokens=2048, top_k=100, eos_token_id=50256)
output = tokenizer.decode(sample[0])
print(output.strip().replace("Assistant:", ""))
