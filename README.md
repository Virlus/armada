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


## ⚙️ Installation

```
conda env create -f conda_environment.yaml
```

If you'd like to test on real robot, execute the following command:

```
conda env create -f conda_environment_real.yaml
```

## 📷 Expert demonstration collection

```
python record.py -o /path/to/data -res {Desired image resolution}
```