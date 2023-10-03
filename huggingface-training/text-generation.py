import warnings
warnings.filterwarnings("ignore", category=Warning)

def generate_text(prompt, max_length=60, num_sequences=1):
  from transformers import pipeline
  generator = pipeline("text-generation",
    # model="distilgpt2"
    model="mistralai/Mistral-7B-v0.1"
  )
  return generator(
    prompt,
    max_length=max_length,
    num_return_sequences=num_sequences,
  )

def main():
  prompt = "In the"
  result = generate_text(prompt)
  print(result)

if __name__ == "__main__":
  main()
