<div align="center">
<h1>ARMADA: Autonomous Online Failure Detection and Human Shared Control Empower Scalable Real-world Deployment and Adaptation</h1>

<a href="https://virlus.github.io/armada/"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
<a href="https://virlus.github.io/armada/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

**Shanghai Jiao Tong University**; **Shanghai Innovation Institute**; **Noematrix Ltd.**

[Wenye Yu](https://virlus.github.io/), [Jun Lv](https://lyuj1998.github.io/), [Zixi Ying](https://github.com/KiriyamaGK), [Yang Jin](https://github.com/EricJin2002), [Chuan Wen](https://alvinwen428.github.io/), [Cewu Lu](https://www.mvig.org/)
</div>


<div align="center">
  <img src="assets/armada_banner.gif" alt="ARMADA banner" width="800"></img>
</div>

## Table of Contents

- [🔥 Highlights](#-highlights)
- [🛠️ Installation](#️-installation)
- [📷 Data Collection](#️-data-collection)
- [🤖 Policy Training](#-policy-training)
- [🌐 Multi-robot deployment](#️-multi-robot-deployment)

## 🔥 Highlights

<p align="center">
<img src="assets/method.pdf" alt="highlights" style="width:90%;" />
</p>

- Our failure detector, FLOAT, achieves **nearly 95% accuracy** in four real-world tasks, improving the SOTA online failure detection approaches by over 20%.
- ARMADA leads to a **more than 4× increase in success rate** and a **greater than 2× decrease in human intervention ratio** compared to previous human-in-the-loop learning approaches that require full-time human supervision.
- ARMADA conduces to saliently larger improvement in task progress and **data efficiency using more robots in parallel**, and **expedites policy adaptation to novel scenarios**.

## 🛠️ Installation

### 💻 Conda Environment

We test our codebase on Python 3.10. Please create an environment named `armada` using the following command. 

```
conda env create -f conda_environment.yaml
```

If you'd like to test on real robot, execute the following command.

```
conda env create -f conda_environment_real.yaml
```

### 🦾 Hardware setup

Please refer to [hardware setup guide](hardware/README.md) for more information.

## 📷 Data Collection

The following example collects expert demonstrations with an image resolution of 224*224 and a 10Hz control frequency.
Please feel free to tailor it to your own needs.

```
python record.py --output /path/to/your/output/path --resolution 224 224 --fps 10
```

## 🤖 Policy Training

The main training code can be found in [aramda/diffusion_policy](./armada/diffusion_policy), and the corresponding configuration files are placed under [armada/config/training](./armada/config/training).

An example usage of our training recipe is in [armada/train.sh](./armada/train.sh).

## 🌐 Multi-robot deployment

