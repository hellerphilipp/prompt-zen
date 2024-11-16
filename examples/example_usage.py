from prompt_zen import Enhancer, EvaluationResult

from langchain_openai import ChatOpenAI
import math

# Initialize the execution model (Can be a smaller model, e.g., GPT-3.5, or Llama3.2-8b)
execution_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Initialize the iteration model used to improve on the prompt (Best to pick a bigger model)
iteration_model = ChatOpenAI(model_name="gpt-4", temperature=0)

# Define the execution function to 
def execution_function(prompt: str) -> str:
    with open("sample_text.txt", "r") as file:
        text = file.read()

    combined_prompt = f"{prompt}\n\nText:\n{text}"

    # Use the LLM to generate a response
    response = execution_model.invoke(combined_prompt)
    return response.content

def evaluator_function(llm_output: str) -> EvaluationResult:
    # Evaluate closeness to 25 characters per chunk
    print("Evaluating...")

    chunks = llm_output.split("\n\n")
    print(f"Chunks: {chunks}")
    character_counts = [len(chunk) for chunk in chunks]
    print(f"Character counts: {character_counts}")
    closeness_to_25 = [abs(25 - count) for count in character_counts]
    print(f"Closeness to 25: {closeness_to_25}")

    def score(num) -> float:
        if num > 1000:
            return 0
        else:
            # Calculate the logarithmic scale
            scaled_value = 10 - (math.log(num + 1, 10) * 3) if num > 0 else 10
            return max(1, min(10, scaled_value))  # Clamp values between 1 and 10


    # Return a score between 0 to 10
    return EvaluationResult(
        output_score = score(sum(closeness_to_25) / len(closeness_to_25)),
        output_feedback = None,
    )

# Initialize the Enhancer
enhancer = Enhancer(
    iteration_model=iteration_model,
    execution_function=execution_function,
    evaluator_function=evaluator_function,
)

# Define parameters for the trial
base_prompt = "Split this text into chunks of 25 characters, separated by double-newlines."
goal_description = "Chunks should be exactly 25 characters long."
runs_per_prompt = 1
iterations = 3
context_mode = "tails"  # Use top and worst results as context

# Run the trial
results = enhancer.run_trial(
    base_prompt=base_prompt,
    goal_description=goal_description,
    runs_per_prompt=runs_per_prompt,
    iterations=iterations,
    context_mode=context_mode,
)

# Display results
print(results)
