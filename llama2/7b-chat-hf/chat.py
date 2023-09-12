# thanks to https://pub.towardsai.net/a-simple-hugging-face-guide-to-chatting-with-the-llama-2-7b-model-in-a-colab-notebook-34f4a7e36e17
import torch
from transformers import AutoTokenizer
from transformers import pipeline

model = "../../models/meta-llama_Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


def get_llama_response(prompt: str) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """

    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=256,
    )
    print("Chatbot:", sequences[0]['generated_text'])

def test():
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
    main()