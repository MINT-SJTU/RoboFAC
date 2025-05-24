# RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://arxiv.org/abs/2505.12224) 
[![Paper](https://img.shields.io/badge/Paper-PDF-red)]()
[![Dataset](https://img.shields.io/badge/Dataset-Huggingface-green)](https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset)
[![Model](https://img.shields.io/badge/Model-Huggingface-yellow)]()

This is the official repo for our RoboFAC paper: "RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction".

## Introduction

RoboFAC is a comprehensive framework for robotic failure analysis and correction. 

- It provides a large-scale and diverse robotic failure QA dataset, covering a wide range of tasks, environments, and viewpoints. It includes eight QA types targeting different aspect of robotic failure understanding and correction.
- It proposes models for robotic failure video understanding, capable of comprehenive task understanding, failure analysis, and failure correction. 
- It includes a benchmark dataset for evaluating the failure understanding and correction performance of robotic failure correction models. 
- The model is integrated into a real-world robotic control pipeline as an external critic, enabling real-time correction for VLA-based systems.

## Contents

- [Data Generation](#data-generation)
- [Models](#models)
- [Benchmark](#benchmark)
- [Real-world Control](#real-world-control)

## Data Generation

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

## Models  
Detailed implementations and pre-trained weights will be released in upcoming updates. Stay tuned!  

## Benchmark  
Comprehensive benchmark results and comparative experiments are being organized, and will be added soon.  

## Real-world Control  
Real-world control interfaces and deployment solutions are currently under development. Documentation and sample code will be released shortly.  

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
