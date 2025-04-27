import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from pipelinerl.sglang_utils import SGLangWeightUpdateManager
from pipelinerl.cot_math_agent import CoTMathAgent, RLMathTape
from pipelinerl.state import TrainerState
from pipelinerl.streams import SingleStreamSpec, set_streams_backend
from pipelinerl.load_datasets import load_datasets
from pipelinerl.utils import setup_logging

# Import SGLang components
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

class SGLangLLM:
    """
    Adapter class that wraps SGLang ModelRunner to provide a compatible interface
    with the PipelineRL TrainableLLM.
    """
    
    def __init__(
        self,
        model_runner: ModelRunner,
        model_name: str,
        tokenizer_name: str,
        parameters: Dict[str, Any],
        collect_logprobs: bool = True,
    ):
        self.model_runner = model_runner
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.parameters = parameters
        self.collect_logprobs = collect_logprobs
        
        # Load tokenizer from the model runner
        self.tokenizer = model_runner.tokenizer
        
    async def generate(self, prompt: str, **kwargs):
        """
        Generate text completion using SGLang ModelRunner.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated completion
        """
        # Set up generation parameters
        params = {
            "temperature": self.parameters.get("temperature", 0.0),
            "top_p": self.parameters.get("top_p", 1.0),
            "max_tokens": self.parameters.get("max_tokens", 512),
        }
        
        # Merge with any kwargs provided
        params.update(kwargs)
        
        # Call SGLang ModelRunner for generation
        response = await self.model_runner.generate_text(
            prompt=prompt,
            **params
        )
        
        # Format the response similar to what TrainableLLM returns
        return {
            "text": response.text,
            "tokens": response.generated_tokens,
            "logprobs": response.logprobs if self.collect_logprobs else None,
        }


def create_model_runners(
    model_path: str,
    model_config: Dict[str, Any],
    num_replicas: int,
) -> List[ModelRunner]:
    """
    Create SGLang ModelRunner instances.
    
    Args:
        model_path: Path to the model
        model_config: Configuration for the model
        num_replicas: Number of replica model runners to create
        
    Returns:
        List of ModelRunner instances
    """
    model_runners = []
    
    # Create specified number of model runners
    for i in range(num_replicas):
        # Configure the server for each replica
        server_args = ServerArgs(
            model=model_path,
            port=10000 + i,  # Use different ports for each replica
            device=f"cuda:{i % torch.cuda.device_count()}",
            **model_config.get("server_args", {})
        )
        
        # Configure the model
        model_config_obj = ModelConfig(
            model=model_path,
            **model_config.get("model_config", {})
        )
        
        # Create the model runner
        model_runner = ModelRunner(
            server_args=server_args,
            model_config=model_config_obj,
        )
        
        model_runners.append(model_runner)
        logger.info(f"Created SGLang ModelRunner replica {i}")
    
    return model_runners


def init_sglang_actors(
    cfg: DictConfig,
    exp_path: Path,
    trainer_state: Optional[TrainerState] = None,
) -> Dict[str, Any]:
    """
    Initialize SGLang actors and weight update manager.
    
    Args:
        cfg: Configuration dictionary
        exp_path: Path to experiment directory
        trainer_state: Optional trainer state
        
    Returns:
        Dictionary with initialized components
    """
    # Load model path
    finetune_model_path = exp_path / "finetune" / "current"
    if os.path.exists(finetune_model_path):
        actor_model_path = str(finetune_model_path)
    else:
        actor_model_path = str(cfg.model_path)
    
    # Create model runners
    num_replicas = cfg.actor.num_replicas
    model_runners = create_model_runners(
        model_path=actor_model_path,
        model_config=cfg.sglang.model_config,
        num_replicas=num_replicas,
    )
    
    # Create SGLang LLM wrappers
    train_llms = [
        SGLangLLM(
            model_runner=runner,
            model_name=actor_model_path,
            tokenizer_name=actor_model_path,
            parameters=cfg.llm.parameters,
            collect_logprobs=True,
        )
        for runner in model_runners
    ]
    
    test_llms = [
        SGLangLLM(
            model_runner=runner,
            model_name=actor_model_path,
            tokenizer_name=actor_model_path,
            parameters=cfg.test_llm.parameters,
            collect_logprobs=True,
        )
        for runner in model_runners
    ]
    
    # Create agent replicas
    train_agent_replicas = [
        CoTMathAgent.create(
            system_prompt=cfg.system_prompt,
            llm=llm,
            max_prompt_length=cfg.agent.max_prompt_length,
        )
        for llm in train_llms
    ]
    
    test_agent_replicas = [
        CoTMathAgent.create(
            system_prompt=cfg.system_prompt,
            llm=llm,
            max_prompt_length=cfg.agent.max_prompt_length,
        )
        for llm in test_llms
    ]
    
    # Create weight update manager if trainer state is provided
    weight_update_manager = None
    if trainer_state is not None:
        weight_update_manager = SGLangWeightUpdateManager(
            model_runners=model_runners,
            master_address=cfg.sglang.master_address,
            master_port=cfg.sglang.master_port,
            rank_offset=cfg.sglang.rank_offset,
            world_size=cfg.sglang.world_size,
            group_name=cfg.sglang.group_name,
            backend=cfg.sglang.backend,
        )
        
        # Initialize the weight update group
        success, msg = weight_update_manager.initialize_weight_update_group()
        if not success:
            logger.error(f"Failed to initialize weight update group: {msg}")
        else:
            logger.info("Weight update group initialized successfully")
    
    return {
        "model_runners": model_runners,
        "train_llms": train_llms,
        "test_llms": test_llms,
        "train_agent_replicas": train_agent_replicas,
        "test_agent_replicas": test_agent_replicas,
        "weight_update_manager": weight_update_manager,
    }
    

def run_sglang_actor(cfg: DictConfig):
    """
    Main entry point for running SGLang-based actors.
    
    Args:
        cfg: Configuration dictionary
    """
    # Set up experiment path and logging
    exp_path = Path(cfg.output_dir)
    setup_logging(str(exp_path / "actor"))
    set_streams_backend(**cfg.streams)
    
    # Load datasets
    train_dataset = load_datasets(cfg.train_dataset_names)
    test_dataset = load_datasets(cfg.test_dataset_names)
    if cfg.train_subset:
        train_dataset = train_dataset[cfg.train_subset.begin:cfg.train_subset.end]
    logger.info(f"Loaded {len(train_dataset)} training problems")
    logger.info(f"Loaded {len(test_dataset)} test problems")
    
    # Create trainer state
    trainer_state = TrainerState(exp_path)
    if cfg.debug.mode in ["actor", "open_loop"]:
        trainer_state.propagated_weight_version = 0
    else:
        trainer_state.start_listening()
        while trainer_state.propagated_weight_version is None:
            logger.info("Waiting for the trainer to declare the initial weight version")
            time.sleep(1)
    
    # Initialize SGLang actors
    actor_context = init_sglang_actors(
        cfg=cfg,
        exp_path=exp_path,
        trainer_state=trainer_state,
    )
    
    # Create streams
    tape_stream = SingleStreamSpec(exp_path=exp_path, topic="tapes")
    test_tape_stream = SingleStreamSpec(exp_path=exp_path, topic="test_tapes")
    stats_stream = SingleStreamSpec(exp_path=exp_path, topic="stats")
    data_stream = SingleStreamSpec(exp_path=exp_path, topic="actor")
    
    # Import ActorLoop from pipelinerl.run_actor
    from pipelinerl.run_actor import ActorLoop
    
    # Create actor loops
    train_set_generator = ActorLoop(
        agent_replicas=actor_context["train_agent_replicas"],
        tapes_stream=tape_stream,
        data_stream=data_stream,
        cfg=cfg,
        trainer_state=trainer_state,
        stats_stream=stats_stream,
    ).run_agent_make_data(
        dataset=train_dataset,
        is_training=True,
    )
    
    test_actor_loop = ActorLoop(
        agent_replicas=actor_context["test_agent_replicas"],
        tapes_stream=test_tape_stream,
        data_stream=None,
        cfg=cfg,
        trainer_state=trainer_state,
        stats_stream=stats_stream,
    )
    
    # Run the main loop
    last_regular_eval = -1
    while True:
        next_regular_eval = (
            trainer_state.propagated_weight_version
            if last_regular_eval == -1
            else last_regular_eval + cfg.eval_every_n_versions
        )
        if (
            cfg.eval_every_n_versions
            and not cfg.debug.mode
            and trainer_state.propagated_weight_version >= next_regular_eval
            and test_dataset
        ):
            logger.info("Evaluating model on test set")
            _ = list(
                test_actor_loop.run_agent_make_data(
                    dataset=test_dataset,
                    is_training=False,
                )
            )
            last_regular_eval = next_regular_eval
        else:
            _ = next(train_set_generator) 