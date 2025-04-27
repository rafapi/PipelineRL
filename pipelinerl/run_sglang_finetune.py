import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from pipelinerl.sglang_utils import SGLangWeightUpdateManager
from pipelinerl.streams import SingleStreamSpec, write_to_streams
from pipelinerl.finetune.context import get_accelerator

# Import SGLang components
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

class SGLangTrainerWeightUpdateManager:
    """
    Manages SGLang-based weight updates from the trainer side.
    This class handles broadcasting model weights from the trainer to actors.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        weight_update_stream: SingleStreamSpec,
        cfg: DictConfig,
    ):
        """
        Initialize the trainer weight update manager.
        
        Args:
            model: The PyTorch model being trained
            optimizer: The optimizer
            weight_update_stream: Stream for weight update notifications
            cfg: Configuration dictionary
        """
        self.model = model
        self.optimizer = optimizer
        self.weight_update_stream = weight_update_stream
        self.cfg = cfg
        
        # Create a model runner for the trainer
        # This is only used for weight updates, not for inference
        server_args = ServerArgs(
            model=cfg.model_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            port=9999,  # Use a different port from actors
        )
        
        model_config = ModelConfig(
            model=cfg.model_path,
        )
        
        self.model_runner = ModelRunner(
            server_args=server_args,
            model_config=model_config,
        )
        
        # Initialize the weight update group
        success, msg = self.model_runner.init_weights_update_group(
            master_address=cfg.sglang.master_address,
            master_port=cfg.sglang.master_port,
            rank_offset=0,  # Trainer is always rank 0
            world_size=cfg.sglang.world_size,
            group_name=cfg.sglang.group_name,
            backend=cfg.sglang.backend,
        )
        
        if not success:
            logger.error(f"Failed to initialize weight update group: {msg}")
        else:
            logger.info("Weight update group initialized successfully")
        
        self.current_version = 0
    
    def update_weights(self, version: Optional[int] = None) -> bool:
        """
        Update weights from trainer to actors.
        
        Args:
            version: Optional version number (defaults to incrementing current version)
            
        Returns:
            True if update successful, False otherwise
        """
        # Update version if not provided
        if version is None:
            self.current_version += 1
            version = self.current_version
        else:
            self.current_version = version
        
        logger.info(f"Sending weight update for version {version}")
        
        # Get the appropriate model for broadcasting
        if hasattr(self.model, "module"):
            # Handle distributed training wrapper cases
            if hasattr(self.model.module, "named_parameters"):
                # For DataParallel-like wrappers
                model_to_broadcast = self.model.module
            else:
                # Fall back to the wrapped model
                model_to_broadcast = self.model
        else:
            # Normal case - single model
            model_to_broadcast = self.model
        
        # Make sure all pending optimizer steps are completed
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        # Make sure the param is on CPU or GPU
                        if not param.grad.is_cuda and not param.is_cpu:
                            param.grad = param.grad.to(param.device)
            
            # Ensure optimizer state is synchronized
            if hasattr(self.optimizer, "synchronize"):
                self.optimizer.synchronize()
        
        # Broadcast weights to all model runners (actors)
        if get_accelerator().is_main_process:
            # Broadcast the parameters one by one using SGLang's API
            all_success = True
            error_messages = []
            
            for name, param in model_to_broadcast.named_parameters():
                if param.requires_grad:
                    # Use the ModelRunner's update_weights_from_distributed method
                    # This will broadcast this parameter to all other ranks
                    success, msg = self.model_runner.update_weights_from_distributed(
                        name=name,
                        dtype=param.dtype,
                        shape=tuple(param.shape),
                    )
                    
                    if not success:
                        all_success = False
                        error_messages.append(f"Update failed for {name}: {msg}")
            
            # Notify that weights have been updated through the update stream
            with write_to_streams(self.weight_update_stream) as writer:
                writer.write({
                    "kind": "weight_update_success",
                    "version": version,
                    "timestamp": time.time()
                })
            
            if all_success:
                logger.info(f"Successfully sent weight update version {version}")
            else:
                logger.error(f"Errors during weight update: {'; '.join(error_messages)}")
            
            return all_success
        
        return False


def create_sglang_trainer_updater(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    exp_path: Path,
) -> SGLangTrainerWeightUpdateManager:
    """
    Create a SGLang trainer weight updater.
    
    Args:
        model: The PyTorch model being trained
        optimizer: The optimizer
        cfg: Configuration dictionary
        exp_path: Path to experiment directory
        
    Returns:
        SGLangTrainerWeightUpdateManager instance
    """
    # Create weight update stream
    weight_update_stream = SingleStreamSpec(exp_path=exp_path, topic="weight_update_request")
    
    # Create trainer weight update manager
    return SGLangTrainerWeightUpdateManager(
        model=model,
        optimizer=optimizer,
        weight_update_stream=weight_update_stream,
        cfg=cfg,
    )


def integrate_sglang_weight_updates(cfg: DictConfig):
    """
    Integration utility for SGLang weight updates in the training script.
    
    This function demonstrates how to integrate SGLang weight updates
    into an existing training script.
    
    Args:
        cfg: Configuration dictionary
    """
    # Example code showing how to add SGLang to your existing training script
    
    # 1. In your training script initialization section:
    # exp_path = Path(cfg.output_dir)
    # model = load_model(...)
    # optimizer = get_optimizer(...)
    # 
    # # Initialize SGLang weight updater
    # sglang_updater = create_sglang_trainer_updater(
    #     model=model,
    #     optimizer=optimizer,
    #     cfg=cfg,
    #     exp_path=exp_path,
    # )
    
    # 2. In your training loop, after optimizer step:
    # optimizer.step()
    # optimizer.zero_grad()
    # 
    # # Check if it's time to update weights
    # if step % cfg.finetune.weight_update_interval == 0:
    #     # Send weights to actors
    #     sglang_updater.update_weights()
    
    logger.info("This is a utility module for SGLang weight update integration.")
    logger.info("Import the functions into your training script to use them.") 