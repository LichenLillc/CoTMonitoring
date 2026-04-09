# CoTMonitoring

This repository contains the codebase for our study of whether synthetic reward-hacking trajectories reflect naturally emerging reward hacking in code generation. It combines two main parts:

- `Trace-and-Amplify_Framework/`: the GRPO-based training framework used to curate in-the-wild reward-hacking trajectories.
- `CoTMonitors_Training_and_Evaluation/`: data preprocessing, activation extraction, hidden-state probe training, text-based monitor training, and monitor evaluation utilities.

## Repository Structure

```text
CoTMonitoring/
├── CoTMonitors_Training_and_Evaluation/
├── Trace-and-Amplify_Framework/
├── requirements.txt
└── README.md
```

## Environment Setup

### Docker Environment

If you prefer to use Docker for the framework, start from the framework-compatible image:

```bash
docker pull whatcanyousee/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te2.0-megatron0.11.0-v0.0.6

docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
  -v /path/to/CoTMonitoring:/workspace/CoTMonitoring \
  whatcanyousee/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te2.0-megatron0.11.0-v0.0.6
```

Inside the container:

```bash
cd /workspace/CoTMonitoring
pip install --upgrade pip
pip install -r requirements.txt
pip install -e Trace-and-Amplify_Framework
python -c "import nltk; nltk.download('punkt')"
```

### Conda Environment

```bash
conda create -n cotmonitoring python=3.10
conda activate cotmonitoring

pip install --upgrade pip
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
pip install -e Trace-and-Amplify_Framework
python -c "import nltk; nltk.download('punkt')"
```

## Trace-and-Amplify_Framework

This component is used to collect in-the-wild reward-hacking trajectories for code-generation experiments.

### Training Data Preparation

Framework-side training data is stored under:

```text
Trace-and-Amplify_Framework/or1_data/
└── train/
```

To prepare the full code training set:

```bash
cd Trace-and-Amplify_Framework
python ./or1_scripts/data_preprocess/download_full_data.py --local_dir ./or1_data/train
# injecting fake unit tests
python ./my_utils/add_fake_test_final-ver.py
```

This writes:

```text
Trace-and-Amplify_Framework/or1_data/train/train_full_code.pkl
Trace-and-Amplify_Framework/or1_data/train/train_full_code_faketest.pkl
```

If you use custom framework-side training data, you could place it under `Trace-and-Amplify_Framework/or1_data/train/` in a format expected by the downstream scripts.

### Evaluation Data Preparation

Framework-side evaluation data is stored under:

```text
Trace-and-Amplify_Framework/or1_data/
└── eval/
```

External benchmarks should be placed in the expected subdirectories before running evaluation. For example, LiveCodeBench-style data should live under:

```text
Trace-and-Amplify_Framework/or1_data/eval/livecodebench/
```

One typical preparation flow is:

```bash
cd Trace-and-Amplify_Framework
huggingface-cli download <dataset-name> --repo-type=dataset --local-dir ./or1_data/eval/livecodebench
unzip ./or1_data/eval/livecodebench/livecodebench.zip -d ./or1_data/eval/livecodebench/
mv ./or1_data/eval/livecodebench/livecodebench/* ./or1_data/eval/livecodebench/
```

Make sure the local directory layout matches the paths expected by the evaluation scripts.

### Training Script

Training is launched by:

```bash
cd Trace-and-Amplify_Framework
bash train_script.sh
```

Before running it, download the required Hugging Face model checkpoints into:

```text
Trace-and-Amplify_Framework/local_models/
```

The script expects `MODEL_PATH=./local_models/` and also uses the evaluation data path for LiveCodeBench.

### Inference for Trajectory Collection

To use a trained reward-hacker checkpoint to generate trajectories, run:

```bash
cd Trace-and-Amplify_Framework
bash ./or1_scripts/eval/eval_1p5b.sh
```

Before running it, set `model.path` in [eval_1p5b.sh](Trace-and-Amplify_Framework/or1_scripts/eval/eval_1p5b.sh) to the checkpoint you want to use. The script writes generated trajectories under:

```text
Trace-and-Amplify_Framework/outputs/evaluation/
```

## CoTMonitors_Training_and_Evaluation

This component contains the monitoring pipeline used to preprocess trajectories, extract hidden states, and train and evaluate various monitors.

### Data Preparation

The monitoring pipeline uses both synthetic trajectories and in-the-wild trajectories curated from the framework side. For framework-generated trajectories, the preparation flow is:

1. Filter the inferred trajectories with [filter_infer_traj.py](CoTMonitors_Training_and_Evaluation/scripts/filter_infer_traj.py).
2. Reformat the filtered trajectories and add labels with [reformat_infer_traj.py](CoTMonitors_Training_and_Evaluation/scripts/reformat_infer_traj.py).

A typical flow is:

```bash
cd CoTMonitors_Training_and_Evaluation
python scripts/filter_infer_traj.py --input_file ../Trace-and-Amplify_Framework/outputs/evaluation/<inference_file>.jsonl --score_split
python scripts/reformat_infer_traj.py -i <filtered_files> -o <output_file>.jsonl --hacking_label 1 --hacking_type <type> --traj_source <source_name> --global_dedup_index
```

Place the resulting labeled trajectory files in the experiment-specific locations expected by the downstream monitoring scripts.

## Running Experiments

The monitoring experiments are organized by monitor type:

### Hidden-State-Based Probe Monitors

- Training: [mine_probe_pipeline_loop_paired.sh](CoTMonitors_Training_and_Evaluation/scripts/paired_probe_pipeline/mine_probe_pipeline_loop_paired.sh)
- Evaluation: [5a_eval_probes_dual_loop_paired_final-recipe.py](CoTMonitors_Training_and_Evaluation/scripts/paired_probe_pipeline/5a_eval_probes_dual_loop_paired_final-recipe.py)

### Text Classifiers

- Training: [BERT_pipeline.sh](CoTMonitors_Training_and_Evaluation/scripts/paired_probe_pipeline/BERT_pipeline.sh)
- Evaluation: [5b_eval_text_classifier_loop.py](CoTMonitors_Training_and_Evaluation/scripts/paired_probe_pipeline/5b_eval_text_classifier_loop.py)

### LLM Monitors

- Evaluation: [gpt_evaluate.py](CoTMonitors_Training_and_Evaluation/scripts/gpt_evaluate.py)

## Acknowledgements

Acknowledgements will be added here. Relevant upstream or related codebases include:

- [Placeholder 1]
- [Placeholder 2]
- [Placeholder 3]
