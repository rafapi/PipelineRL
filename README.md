Pipeline RL
=================

# Installation

Clone the repository and change the directory `pipelinerl`
```bash
git clone git@github.com:ServiceNow/research-now-reasoner.git
cd pipelinerl
```

Create the environments with dependencies.
```bash
conda create -n pipeline-rl -y python=3.11
conda run --no-capture-output -n pipeline-rl pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 
conda run --no-capture-output -n pipeline-rl pip install -r requirements.txt --no-build-isolation
```

By default Pipeline-RL will use the file system as the medium for streaming the generated data to the trainer processes. This works on one node, but the files can get pretty large. To use Redis instead you will need to install the Redis server in the same conda environment:
```bash
conda install redis-server==7.4.0 -c conda-forge 
```

# Run experiments

First, activate the conda environment:
```bash
conda activate pipeline-rl
```

Single node with 8 H00 GPUs:

```bash
python -m pipelinerl.launch output_dir=results/base1
```

If you only have 4 H100 GPUs:
```bash
python -m pipelinerl.launch --config-name base_4gpu output_dir=results/base1 
```

To use Redis instead of the filesystem for data streaming:
```
python -m pipelinerl.launch streams=redis output_dir=results/base1
```

Multi node: coming soon.