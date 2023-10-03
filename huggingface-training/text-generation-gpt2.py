import warnings
warnings.filterwarnings("ignore", category=Warning)

def generate_text(prompt, max_length=60, num_sequences=10):
  from transformers import pipeline
  generator = pipeline("text-generation",
    model="distilgpt2"
  )

  return generator(
    prompt,
    max_length=max_length,
    num_return_sequences=num_sequences,
  )

def main():
  prompt = "APIKEY="
  results = generate_text(prompt)
  for result in results:
    print(f'{result}')

if __name__ == "__main__":
  main()