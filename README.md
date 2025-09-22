# Code In Place AI Evaluation Project

This repository accompanies the paper:  
**“Aligning Small Language Models for Programming Feedback: Towards Scalable Coding Support in a Massive Global Course”**  
(*to appear in the proceedings of SIGCSE TS 2026*).  

👉 Project page: [https://koutche.me/cip25-aiep/](https://koutche.me/cip25-aiep/)  

---

## Overview

This repository contains the core code to **train small language models (SLMs)** for **rubric-based programming feedback**, as presented in the paper.  
It includes:
- Training scripts for supervised and preference optimization.  
- A sample pipeline that reproduces the training flow described in the paper.  
- Environment specification for reproducibility.  

⚠️ Due to privacy restrictions, **data preprocessing scripts** and **human evaluation code** are not included.  
However, the pipeline can be easily adapted to your own dataset.

---

## Installation

Setup the environment:

```bash
conda env create --name diag --file=environment.yaml
conda activate diag
```

We also provide a sample script (runnable on HPC systems) that illustrates the training and validation process:

```bash
bash scripts/pipe.sh
```

Feel free to contact (charles.koutcheme@aalto.fi) for support in setting up the pipeline. 

## Citation 

Official DOI will be available early next year.

```
@inproceedings{Koutcheme2026CIP,
  author    = {Charles Koutcheme and Juliette Woodrow and Chris Piech},
  title     = {Aligning Small Language Models for Programming Feedback: Towards Scalable Coding Support in a Massive Global Course},
  booktitle = {Proceedings of the 57th ACM Technical Symposium on Computer Science Education (SIGCSE TS)},
  year      = {2026}
}
```