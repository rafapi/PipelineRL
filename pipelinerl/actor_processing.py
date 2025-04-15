import logging
from typing import Dict, List, Tuple

from omegaconf import DictConfig

import requests
from datasets import load_dataset
from tqdm import tqdm
from tapeagents.core import LLMCall, StepMetadata, TrainingText
from pipelinerl.finetune.data import MASKED_TOKEN_ID
from pipelinerl.verifier_api import verify_answer
from .cot_math_agent import CoTMathAgent, RLMathTape, Task

from pydantic import BaseModel
logger = logging.getLogger(__name__)

class RewardTable(BaseModel):
    wrong_answer_not_finished: float
    wrong_answer_finished: float
    no_answer_not_finished: float
    no_answer_finished: float
    unparsable_not_finished: float
    unparsable_finished: float
    correct_answer_not_finished: float
    correct_answer_finished: float

def convert_problems_to_tapes(problems: list, cfg: DictConfig) -> list[RLMathTape]:
    """
    Creates RLMathTape objects from a list of math problem dictionaries.

    Args:
        problems (list[dict]): List of dictionaries containing math problems, where each dict
            has 'question' and expected answer value. The list is created from a dataset.

    Returns:
        list[RLMathTape]: List of RLMathTape objects initialized with the math problems as Task steps.
            Each tape contains a single starting Task step with the question and expected answer value
            stored in metadata.
    """
    tapes: list[RLMathTape] = []
    for problem in problems:
        start_step = Task(
            task=problem["task"],
            template=cfg.task_template,
            metadata=StepMetadata(
                other={
                    "answer": problem["answer"],
                    "dataset": problem.get("dataset", ""),
                    "id": problem["id"],
                }
            ),
        )
        tape = RLMathTape(steps=[start_step], context=None)
        tapes.append(tape)
    return tapes




def extract_tape_training_samples(
    new_tape: RLMathTape, agent: CoTMathAgent, cfg: DictConfig
) -> Tuple[List[TrainingText], Dict[str, int]]:
    """
    Process a single tape to extract training samples and statistics.

    Args:
        new_tape: The tape to process containing math problem steps
        agent: CoTMathAgent
        tapes_dir: Directory to save processed tapes
        cfg: Configuration

    Returns:
        Tuple containing:
        - List of training samples with rewards and logprobs
        - Dictionary with statistics (reward, steps, success, no_errors)
    """
    tape_prompt_tokens = 0
    tape_output_tokens = 0

    prediction = new_tape.steps[-1].reasoning
    gold_answer = new_tape.steps[0].metadata.other["answer"]
    answer_status = verify_answer(
        prediction=prediction,
        gold=gold_answer, 
        strict=True, 
    )

    training_samples: list[TrainingText] = []
    # For each LLM interaction in the tape:
    # - Create a training sample from the prompt and output
    # - Get log probabilities of the output tokens
    # - Set group ID for tracking
    reward = overflows = None
    for step in new_tape.steps:
        if (
            "llm_call" not in step.metadata.other
            or step.metadata.other["llm_call"] is None
        ):
            continue
        llm_call = step.metadata.other["llm_call"]

        if isinstance(llm_call, dict):
            llm_call = LLMCall(**llm_call)

        tape_prompt_tokens += llm_call.prompt_length_tokens
        tape_output_tokens += llm_call.output_length_tokens
        overflows = []
        trace = agent.llm.make_training_text(llm_call.prompt, llm_call.output)

        input_ids = [lp.token_id for lp in llm_call.logprobs]
        labels = [lp.token_id for lp in llm_call.logprobs if lp.generated]
        # finished can be 0, 1
        finished = 1 if input_ids[-1] == agent.llm.tokenizer.eos_token_id else 0
        rewards = RewardTable(**dict(cfg.rewards))

        match (answer_status, finished):
            case ("wrong", 0):
                reward = rewards.wrong_answer_not_finished
            case ("wrong", 1):
                reward = rewards.wrong_answer_finished
            case ("no_answer", 0):
                reward = rewards.no_answer_not_finished
            case ("no_answer", 1):
                reward = rewards.no_answer_finished
            case ("unparsable", 0):
                reward = rewards.unparsable_not_finished
            case ("unparsable", 1):
                reward = rewards.unparsable_finished
            case ("correct", 0):
                reward = rewards.correct_answer_not_finished
            case ("correct", 1):
                reward = rewards.correct_answer_finished
            case _:
                raise ValueError(f"Invalid success/finished combination: {answer_status}{finished}")
        reward *= cfg.discount_factor ** llm_call.output_length_tokens

        # MASKED_TOKEN_ID is -100 and is the default "ignore_index" in nn.CrossEntropyLoss,
        # see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        labels = [MASKED_TOKEN_ID] * (len(input_ids) - len(labels)) + labels

        trace.input_ids = input_ids
        trace.labels = labels
        trace.reward = reward

        # check if the last produced token is the end of sequence token
        overflows.append(not finished)
        trace.logprobs = [lp.logprob for lp in llm_call.logprobs if lp.generated]
        trace.group_id = new_tape.metadata.parent_id
        training_samples.append(trace)

        # Also ad reward info to the tape 
        step.metadata.other = {
            "reward": reward,
            "answer_status": answer_status,
            "finished": finished,
        }
    assert reward is not None and overflows is not None

    tape_stats = {
        "reward": reward,
        "steps": len(new_tape.steps),
        "success": answer_status == "correct",
        "no_error": answer_status != "unparsable",
        "no_answer": answer_status == "no_answer",
        "prompt_tokens": tape_prompt_tokens,
        "output_tokens": tape_output_tokens,
        "overflows": sum(overflows),
    }
    return training_samples, tape_stats

