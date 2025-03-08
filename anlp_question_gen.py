import os
import litellm

def generate_questions(chunk):
    """
    Generate questions from a given text chunk using an LLM.
    """
    response = litellm.completion(
        api_key=os.environ.get("LITELLM_API_KEY"),
        base_url="https://cmu.litellm.ai",
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates questions from a given text."},
            {"role": "user", "content": f"Generate a list of three relevant questions from the following text: {chunk}"}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Example usage
chunk = "The Los Angeles Dodgers won the World Series in 2020. The games were played in Arlington, Texas, at Globe Life Field due to the COVID-19 pandemic."
questions = generate_questions(chunk)
print(questions)