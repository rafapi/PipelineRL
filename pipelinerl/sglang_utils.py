import logging
import time
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

# Import from SGLang
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import init_custom_process_group

logger = logging.getLogger(__name__)

class SGLangWeightUpdateManager:
    """
    Manages weight updates using SGLang's ModelRunner API for distributed
    weight broadcasting from the trainer (rank 0) to actors (ranks 1+).
    """
    
    def __init__(
        self,
        model_runners: List[ModelRunner],
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str = "actor_update",
        backend: str = "nccl",
    ):
        """
        Initialize the weight update manager.
        
        Args:
            model_runners: List of SGLang ModelRunner instances to update
            master_address: IP address of the master node (trainer)
            master_port: Port for NCCL communication
            rank_offset: Rank offset (0 for trainer, 1+ for actors)
            world_size: Total number of processes in the update group
            group_name: Name for the process group
            backend: Communication backend ("nccl" recommended for GPUs)
        """
        self.model_runners = model_runners
        self.master_address = master_address
        self.master_port = master_port
        self.rank_offset = rank_offset
        self.world_size = world_size
        self.group_name = group_name
        self.backend = backend
        self.thread_pool = ThreadPoolExecutor(max_workers=len(model_runners))
        self.initialized = False
        
    def initialize_weight_update_group(self) -> Tuple[bool, str]:
        """
        Initialize the NCCL process group for weight updates.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        success_results = []
        error_messages = []
        
        # Initialize process group on each model runner
        futures = []
        for i, runner in enumerate(self.model_runners):
            future = self.thread_pool.submit(
                runner.init_weights_update_group,
                master_address=self.master_address,
                master_port=self.master_port,
                rank_offset=self.rank_offset + i,  # Offset rank for each model runner
                world_size=self.world_size,
                group_name=self.group_name,
                backend=self.backend,
            )
            futures.append(future)
        
        # Wait for all initializations to complete
        for future in futures:
            success, msg = future.result()
            success_results.append(success)
            if not success:
                error_messages.append(msg)
        
        all_successful = all(success_results)
        if all_successful:
            self.initialized = True
            return True, "Weight update group initialized successfully"
        else:
            error_msg = "; ".join(error_messages)
            return False, f"Failed to initialize weight update group: {error_msg}"

    def update_weights_from_distributed(self, params_info: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Update weights from distributed parameters (for actors).
        
        Args:
            params_info: List of parameter info dictionaries with name, dtype, and shape
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.initialized:
            return False, "Weight update group not initialized"
        
        success_results = []
        error_messages = []
        
        # Update weights for each parameter in each model runner
        for info in params_info:
            futures = []
            for runner in self.model_runners:
                future = self.thread_pool.submit(
                    runner.update_weights_from_distributed,
                    name=info["name"],
                    dtype=info["dtype"],
                    shape=info["shape"],
                )
                futures.append(future)
            
            # Wait for this parameter to be updated in all model runners
            for future in futures:
                success, msg = future.result()
                success_results.append(success)
                if not success:
                    error_messages.append(f"{info['name']}: {msg}")
        
        all_successful = all(success_results)
        if all_successful:
            return True, "Weights updated successfully"
        else:
            error_msg = "; ".join(error_messages)
            return False, f"Failed to update weights: {error_msg}"
            
    def broadcast_model_weights(self, model: torch.nn.Module) -> Tuple[bool, str]:
        """
        Broadcast model weights from trainer to actors.
        
        Args:
            model: PyTorch model whose weights will be broadcasted
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.initialized:
            return False, "Weight update group not initialized"
        
        success_results = []
        error_messages = []
        
        # Broadcast each parameter
        for name, param in model.named_parameters():
            if param.requires_grad:  # Only broadcast trainable parameters
                futures = []
                for runner in self.model_runners:
                    future = self.thread_pool.submit(
                        runner.update_weights_from_distributed,
                        name=name,
                        dtype=param.dtype,
                        shape=tuple(param.shape),
                    )
                    futures.append(future)
                
                # Wait for this parameter to be updated in all model runners
                for future in futures:
                    success, msg = future.result()
                    success_results.append(success)
                    if not success:
                        error_messages.append(f"{name}: {msg}")
        
        all_successful = all(success_results)
        if all_successful:
            return True, "Model weights broadcasted successfully"
        else:
            error_msg = "; ".join(error_messages)
            return False, f"Failed to broadcast model weights: {error_msg}" 