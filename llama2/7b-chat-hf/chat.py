
def hf_login(token):
    """
    Login to the HuggingFace Hub.
    """
    from huggingface_hub import login
    login(token=token, add_to_git_credential=True)


def get_llama_response(prompt: str) -> None:
    import torch
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=access_token)
    model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=access_token)

    pipeline = transformers.pipeline("text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device = torch.device('mps', index=0)
    )

    sequences = pipeline("what is the recipe of mayonnaise?",
        temperature=0.9,
        top_k=50,
        top_p=0.9,
        max_length=500
    )

    for seq in sequences:
        print(seq['generated_text'])


def test():
    import os
    HF_TOKEN = os.environ.get("HF_TOKEN")
    hf_login(HF_TOKEN)

    prompt = 'how are you?\n'
    get_llama_response(prompt)


def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "quit", "exit"]:
            print("Chatbot: Goodbye!")
            break
        get_llama_response(user_input)

if __name__ == "__main__":
    test()
    #main()
