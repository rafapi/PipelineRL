# SGLang Integration for PipelineRL

This document describes how to integrate SGLang into PipelineRL as a replacement for vLLM, with a particular focus on the weight update mechanism.

## Overview

The integration replaces vLLM's custom weight update mechanism with SGLang's built-in distributed weight update capabilities. SGLang offers cleaner interfaces for weight updates with its `init_weights_update_group` and `update_weights_from_distributed` methods.

## Key Components

The integration consists of these main components:

1. **`SGLangWeightUpdateManager`**: Core class that manages weight updates between trainer and actors
2. **`SGLangLLM`**: Adapter class that wraps SGLang's ModelRunner to provide a compatible interface
3. **`run_sglang_actor.py`**: Implementation for running actors with SGLang
4. **`run_sglang_finetune.py`**: Integration utilities for the trainer side

## Weight Update Mechanism

SGLang allows real-time weight updates through its built-in mechanisms:

### 1. Process Group Initialization

Before any updates, both trainer (rank 0) and actors (ranks 1…N) must join the same custom NCCL group:

```python
success, msg = model_runner.init_weights_update_group(
    master_address="trainer.ip.or.hostname",
    master_port=12345,
    rank_offset=0,               # 0 for trainer, 1+ for actors
    world_size=1 + num_actors,   # total ranks in the group
    group_name="actor_update",   # any identifier for the group
    backend="nccl",
)
```

Under the hood, this initializes a PyTorch distributed process group and creates a barrier to synchronize all ranks.

### 2. Parameter Broadcasting

During training, after each optimizer step (or at specified intervals), the trainer broadcasts updated weights:

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        success, msg = model_runner.update_weights_from_distributed(
            name=name,
            dtype=param.dtype,
            shape=tuple(param.shape),
        )
        assert success, f"Update failed for {name}: {msg}"
```

This performs:
1. Memory allocation on all ranks
2. Broadcasting from rank 0 to all other ranks using `torch.distributed.broadcast`
3. Atomic weight swapping with SGLang's reader-writer lock mechanism

## Why the Complication?

The apparent complexity of SGLang's weight update API serves several important purposes:

1. **Encapsulation of Concurrency Control**: SGLang handles thread safety internally through reader-writer locks, ensuring that weight updates don't interfere with ongoing inference.

2. **Clean Separation of Concerns**: 
   - The `init_weights_update_group` method focuses on process group setup
   - The `update_weights_from_distributed` method handles both communication and atomic tensor swapping

3. **Safety Guarantees**:
   - SGLang ensures parameters are the correct shapes and dtypes before broadcasting
   - It handles synchronization between ranks automatically
   - It manages atomic updates to prevent race conditions with inference

4. **Error Handling**: The API returns success/error status with descriptive messages rather than throwing exceptions, making it more robust in production environments.

In contrast, the manual approach would require:
```python
# Manually create process groups
dist.init_process_group(...)

# Manually broadcast weights with no built-in safety checks
for name, param in model.named_parameters():
    dist.broadcast(param.data, src=0)
    
# Manually handle race conditions with inference
# Manually handle errors and edge cases
```

## Concurrency Control

SGLang provides built-in concurrency control:

- Weight updates acquire a **write lock**, ensuring exclusive access
- Inference requests acquire a **read lock**, allowing concurrent inference
- This prevents race conditions during weight updates

Under the hood, SGLang's ModelRunner uses an asyncio-compatible reader-writer lock:

```python
async with self.model_update_lock.writer_lock():
    # Broadcast and update weights here
    # All inference requests are paused during this time
```

## Implementation Details

### Configuration

SGLang support is built directly into the base configuration system. The `conf/base.yaml` file includes all settings for SGLang:

```yaml
# Flag to control whether to use SGLang instead of vLLM
use_sglang: false

# SGLang-specific configuration
sglang:
  master_address: "${env:MASTER_ADDR,localhost}"
  master_port: ${world.actor_group_port}
  group_name: "actor_update"
  backend: "nccl"
  model_config:
    model_config:
      tensor_parallel_size: 1
      max_model_len: 8192
    server_args:
      max_num_seqs: 128
      quantization: null
```

To enable SGLang, simply override the `use_sglang` flag at runtime.

### Trainer Side Implementation

The trainer initializes a ModelRunner (not for inference, just for weight updates) and uses it to broadcast parameters:

```python
# Initialize the weight update process group
model_runner = ModelRunner(...)
success, msg = model_runner.init_weights_update_group(...)

# In the training loop, after optimizer step:
for name, param in model.named_parameters():
    if param.requires_grad:
        success, msg = model_runner.update_weights_from_distributed(
            name=name,
            dtype=param.dtype,
            shape=tuple(param.shape),
        )
```

### Actor Side Implementation

Actors use SGLang's ModelRunner for both inference and receiving weight updates:

```python
# Initialize actors and join the same process group
model_runners = [ModelRunner(...) for _ in range(num_replicas)]
for runner in model_runners:
    runner.init_weights_update_group(...)

# Weight updates happen automatically when the trainer broadcasts
```

## Benefits Over vLLM's Approach

1. **Cleaner Separation of Concerns**: SGLang cleanly separates loading logic from coordination logic
2. **Built-in Concurrency Control**: Reader-writer locks manage concurrent access automatically
3. **Zero Downtime**: Ongoing inference can complete while new requests wait for weight updates
4. **No Custom Communication Code**: Uses PyTorch's distributed communication directly
5. **Simplified Integration**: Fewer moving parts and cleaner interfaces

## Running the Integration

The SGLang integration is fully integrated into the PipelineRL orchestrator. To use it:

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Enable SGLang when launching the pipeline:
   ```
   python -m pipelinerl.launch +use_sglang=true
   ```

3. You can also customize SGLang settings when needed:
   ```
   python -m pipelinerl.launch +use_sglang=true sglang.model_config.model_config.max_model_len=4096
   ```

4. Monitor training progress through the usual PipelineRL mechanisms.

## Debugging

If you encounter issues with weight updates:

1. Check that all nodes can communicate over the specified port
2. Verify that NCCL is properly installed and configured
3. Ensure SGLang version is at least 0.1.11
4. Check CUDA versions are consistent across all nodes

## Additional Resources

- [SGLang Documentation](https://github.com/sgl-project/sglang)
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html) 