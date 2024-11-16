# prompt-zen

**Prompt-Zen** is a python framework for cultivating and nurturing LLM-prompts, much like tending to a zen garden.

## Features

- **Automatic prompt refinement**: Refine prompts using LangChain-compatible LLMs and feedback.
- **Custom Evaluation**: Incorporate your own scoring and feedback logic for flexible evaluation.
- **Context Sharing**: Share insights from prior generations to inform future iterations.
- **Human-Readable Summaries**: Summarize prompt performance and results in a structured DataFrame.

## Process

1. **Planting the Seed**: Starting with a simple base prompt that describes the goal of the task.
2. **Growing and reaping the fruits**: Using an execution model to run the prompt and generate its initial response.
3. **Pruning and Shaping**: Using an iteration model to provide feedback and measure how well the response aligns with the desired outcome.
4. **Iterative Growth**: Refining the prompt across multiple generations, incorporating lessons learned from top-performing and underperforming results to optimize the output.
5. **Achieving Harmony**: Presenting a clear, effective summary of prompt-performance and a prompt that consistently achieves the desired result—a balanced, well-cultivated interaction.

## Installation

To install `prompt-zen` directly from GitHub, use:

```bash
pip install prompt-zen
```

---

## Usage

Use `prompt-zen` in Jupyter notebooks or Python projects to optimize and refine LLM prompts.

---

## License

This project is licensed under the BSD-3-Clause License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Feel free to open issues, suggest features, or submit pull requests.

---

## Acknowledgments

This project leverages [LangChain](https://github.com/langchain-ai/langchain) for seamless integration with language models.