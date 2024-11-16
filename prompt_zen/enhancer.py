from typing import Callable
from dataclasses import dataclass
import asyncio

from langchain_core.language_models import BaseLLM
import pandas as pd

import traceback
import logging

logger = logging.getLogger(__name__)

@dataclass
class IterationResults:
    iteration: int
    run: int
    prompt: str
    output: str
    output_score: float
    output_feedback: str | None

@dataclass
class EvaluationResult:
    output_score: float
    output_feedback: str | None

class Enhancer:
    """
    A class to refine and optimize prompts for LLM-driven tasks using evaluation and feedback loops.
    """

    def __init__(
        self,
        iteration_model: BaseLLM,
        execution_function: Callable[[str], str],
        evaluator_function: Callable[[str], EvaluationResult],
    ) -> None:
        """
        Initializes the Enhancer class with the required components.

        Args:
            iteration_model (BaseLLM): A LangChain-compatible LLM used to refine prompts.
            execution_function (Callable[[str], str]): A function that takes a base prompt (as a string)
                and executes it, returning the LLM response string.
            evaluator_function (Callable[[str], EvaluationResult]): A function
                that evaluates the output of the execution function against the intended goal and returns
                an EvaluationResult object.
        """
        self.evaluation_model = iteration_model
        self.execution_function = execution_function
        self.evaluator_function = evaluator_function

    def _get_tail_context(
        self,
        full_context: list[IterationResults],
    ) -> list[IterationResults]:
        if not full_context:
            return []
    
        if len(full_context) <= 3:
            return full_context

        sorted_context = sorted(full_context, key=lambda result: result.output_score)

        worst_result = sorted_context[0]
        best_results = sorted_context[-2:]

        return [worst_result] + best_results

    def _create_context_str(
            self,
            context: list[IterationResults],
    ) -> str:
        formatted_entries = []

        for entry in context:
            formatted_entry = f"Prompt: {entry.prompt}\nScore: {entry.output_score:.1f}/10"
            if entry.output_feedback:
                formatted_entry += f"\nFeedback: {entry.output_feedback}"

            formatted_entries.append(formatted_entry)

        return "\n\n---\n\n".join(formatted_entries)

    async def _refine_prompt(
            self,
            current_prompt: str,
            goal_description: str | None,
            context: str
    ) -> str:
        iteration_prompt = """
        You are an expert in crafting effective prompts for language models. Your task is to refine the current prompt to maximize clarity, task alignment, and likelihood of achieving the desired outcome. Use the information below, which includes the current prompt, historical prompts with their scores and feedback, and an explanation of the task goal.

        ### Current Prompt:
        {current_prompt}

        {goal_description}

        ### Historical Prompts and Evaluations:
        {context}

        ### Guidelines:
        1. Focus on improving clarity and specificity.
        2. Ensure the refined prompt aligns with the task goal described above.
        3. Address weaknesses or incorporate feedback from historical prompts (if relevant).
        4. Maintain brevity and avoid unnecessary conversational overhead or ambiguity.
        5. Highlight the changes made, explaining why they enhance the prompt.

        ### Output Format:
        JUST OUTPUT THE REFINED PROMPT - OUTPUT NOTHING MORE.
        """.format(
            current_prompt = current_prompt,
            goal_description = f"### Task Goal:\n{goal_description}" if goal_description else "",
            context = context,

        )
        logger.info("Using iterator_model to improve existing prompt...")
        logger.debug(f"Feeding prompt: {iteration_prompt}")
        llm_result = await self.evaluation_model.ainvoke(iteration_prompt)
        new_prompt = llm_result.content
        logger.info("Produced new prompt {new_prompt} from iteration")
        return new_prompt

    async def run_trial_async(
        self,
        base_prompt: str,
        goal_description: str | None = None,
        runs_per_prompt: int = 1,
        iterations: int = 1,
        context_mode: str = "tails",
    ) -> pd.DataFrame:
        """
        Asynchronously runs a trial to optimize a given prompt.

        Args:
            base_prompt (str): The base prompt to be refined.
            goal_description (str, optional): A short description of the intended goal or outcome.
            runs_per_prompt (int): The number of executions per prompt iteration.
            iterations (int): The number of refinement iterations to perform.
            context_mode (str): Specifies how context is shared with the model:
                - "tails": Only top and worst performers are shared.
                - "full": Full context of what worked and what didn't is shared.

        Returns:
            pd.DataFrame: A dataframe containing trial results and evaluations for each prompt iteration.
        """
        results: list[IterationResults] = []
        prompt: str = base_prompt 

        for iteration in range(iterations):
            logger.info(f"Running iteration {iteration}")
            iteration_results: list[IterationResults] = []

            for run in range(runs_per_prompt):
                logger.info(f"Start of repetition {run}")
                try:
                    output = self.execution_function(prompt)

                    evaluation = self.evaluator_function(output)

                    iteration_results.append(IterationResults(
                        iteration = iteration,
                        run = run,
                        prompt = prompt,
                        output = output,
                        output_score = evaluation.output_score,
                        output_feedback = evaluation.output_feedback,
                    ))
                except:
                    logger.error(f"Execution of iteration {iteration} run {run} failed")
                    logger.error(traceback.format_exc())
                    iteration_results.append(IterationResults(
                        iteration = iteration,
                        run = run,
                        prompt = prompt,
                        output = "",
                        output_score = 0.0,
                        output_feedback = "Execution failed.",
                    ))

            if context_mode == "tails":
                context = self._get_tail_context(iteration_results)
            elif context_mode == "full":
                context = iteration_results
            else:
                context = []

            prompt = await self._refine_prompt(
                current_prompt = prompt,
                goal_description = goal_description,
                context = self._create_context_str(context),
            )

            results.extend(iteration_results)
        
        return pd.DataFrame([result.__dict__ for result in results])

    def run_trial(
        self,
        base_prompt: str,
        goal_description: str | None = None,
        runs_per_prompt: int = 1,
        iterations: int = 10,
        context_mode: str = "tails",
    ) -> pd.DataFrame:
        """
        Runs a trial to optimize a given prompt.

        Args:
            base_prompt (str): The base prompt to be refined.
            goal_description (str, optional): A short description of the intended goal or outcome.
            runs_per_prompt (int): The number of executions per prompt iteration.
            iterations (int): The number of refinement iterations to perform.
            context_mode (str): Specifies how context is shared with the model:
                - "tails": Only top and worst performers are shared.
                - "full": Full context of what worked and what didn't is shared.

        Returns:
            pd.DataFrame: A dataframe containing trial results and evaluations for each prompt iteration.
        """
        return asyncio.run(
            self.run_trial_async(
                base_prompt,
                goal_description,
                runs_per_prompt,
                iterations,
                context_mode,
            )
        )