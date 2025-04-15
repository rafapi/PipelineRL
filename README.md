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
conda run --no-capture-output -n pipeline-rl pip install -r requirements.txt --no-build-isolation
```

You don't have to use Conda, any other virtual environment manager will do.

# Run experiments

Single node with 8 H00 GPUs:

```
conda activate pipeline-rl
python -m pipelinerl.launch output_dir=results/base1
```

If you only have 4 H100 GPUs:
```
conda activate pipeline-rl
python -m pipelinerl.launch --config-name base_4gpu output_dir=results/base1 
```

Multi node: coming soon.