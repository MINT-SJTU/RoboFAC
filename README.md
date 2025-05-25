# RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction

[![Project Page](https://img.shields.io/badge/Project-Page-blue)]() 
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/abs/2505.12224)
[![Dataset](https://img.shields.io/badge/Dataset-Huggingface-green)](https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset)
[![Model](https://img.shields.io/badge/Model-Huggingface-yellow)](https://huggingface.co/MINT-SJTU/RoboFAC-7B)

This is the official repo for our RoboFAC paper: "RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction".

## Introduction
![Overview](./assets/overview.jpg)

RoboFAC is a comprehensive framework for robotic failure analysis and correction. 

- It provides a large-scale and diverse robotic failure QA dataset, covering a wide range of tasks, environments, and viewpoints. It includes eight QA types targeting different aspect of robotic failure understanding and correction.
- It proposes models for robotic failure video understanding, capable of comprehenive task understanding, failure analysis, and failure correction. 
- It includes a benchmark dataset for evaluating the failure understanding and correction performance of robotic failure correction models. 
- The model is integrated into a real-world robotic control pipeline as an external critic, enabling real-time correction for VLA-based systems.

## Contents

- [Dataset](#dataset)
- [Model](#models)
- [Benchmark](#benchmark)
- [Data Generation (Optional)](#data-generation)

## Dataset
![Dataset](./assets/dataset.jpg)
RoboFAC Dataset is a large-scale, high-diversity dataset featuring videos of robot manipulation failures, annotated with QA pairs for comprehensive failure reasoning.
👉 Download from HuggingFace: [The RoboFAC Datset](https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset)

## Model  
![Model result](./assets/result.jpg)
We release RoboFAC-7B, a large vision-language model finetuned for robotic failure analysis and correction.
👉 Get the model from HuggingFace: [The RoboFAC-7B](https://huggingface.co/MINT-SJTU/RoboFAC-7B)

## Evaluation
To evaluate your model's capacity of task understanding, failure analysis and failure correction, simply run the `evaluation/eval.py`.

To speed up the evaluation process, we use the vLLM engine for model inference. Specifically, you need to first run the following command to launch the model:
```bash
python -m vllm.entrypoints.openai.api_server --model YOUR_MODEL_PATH
```
The final results will be in `results_merged.json`, it contains the model's performance scores on each dimension for each robotic task.

## Data Generation (Optional)
We recommend using the provided RoboFAC dataset directly. However, if you want to replicate or extend the data collection process, follow these steps:
### 1. Environment Setup

#### 1.1. Create a Virtual Environment

```bash
# Clone the RoboFAC repository
git clone https://github.com/MINT-SJTU/RoboFAC.git
cd RoboFAC

# Create and activate a conda environment
conda create -n robofac python=3.10 -y
conda activate robofac
```

#### 1.2. Install ManiSkill

Please follow the official [ManiSkill installation guide](https://github.com/haosulab/ManiSkill?tab=readme-ov-file#installation) to set up the simulation environment properly.

#### 1.3. Download Required Scene Assets

We use two simulator environments, `ReplicaCAD` and `AI2THOR`, which are officially supported by ManiSkill. You can download them using the following commands:

```bash
python -m mani_skill.utils.download_asset ReplicaCAD
python -m mani_skill.utils.download_asset AI2THOR
```

#### 1.4. Apply RoboFAC Configuration

After setting up the environment and downloading the assets, run the following script to configure necessary file replacements:

```bash
cd RoboFAC
python setup_config.py
```

### 2. Collect Successful Trajactories
(Optional) To collect successful demonstration trajectories for reference:
```
cd RoboFAC/data_collection
python collect_success.py
```
⚠️ Note: These scripts are provided for reproducibility only and are not required for using the dataset or models.

## Acknowledgement

We thank the following projects that parts of our code are derived from:

- [Maniskill](https://github.com/haosulab/ManiSkill)

## Citation

```bibtex
@misc{lu2025robofaccomprehensiveframeworkrobotic,
      title={RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction}, 
      author={Weifeng Lu and Minghao Ye and Zewei Ye and Ruihan Tao and Shuo Yang and Bo Zhao},
      year={2025},
      eprint={2505.12224},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.12224}, 
}
```
