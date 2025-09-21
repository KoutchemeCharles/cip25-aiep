---
layout: default
---


# Aligning Small Language Models for Programming Feedback

Welcome to the project repository for the **Code In Place AI Evaluation Project (CIP-AIEP)**.  
This project led to the paper:

**Aligning Small Language Models for Programming Feedback:  
Towards Scalable Coding Support in a Massive Global Course**

_To appear in the proceedings of [SIGCSE TS 2026](https://sigcse2026.sigcse.org/), St. Louis, Missouri._

<div style="margin-top:1em; display:flex; gap:1em; flex-wrap:wrap;">
  <a href="https://koutche.me/files/sigcse26_rubric_feedback.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper PDF">
  </a>
  <a href="https://github.com/KoutchemeCharles/cip25-aiep">
    <img src="https://img.shields.io/badge/Code-GitHub-blue" alt="GitHub Code">
  </a>
</div>

---

## Code In Place

[Code In Place](https://codeinplace.stanford.edu/) is a Massive Open Online Course (MOOC) that teaches thousands of learners worldwide the fundamentals of Python programming.  

In this project, we trained **3B-parameter small language models (SLMs)** to provide **diagnostic feedback** on students’ submissions to exam-like programming exxercises.  The models were guided by **rubric-based prompting**, supervised fine-tuning, and preference-based optimization.

---

## Highlights

- ✅ Deployed in **Code In Place 2025**, a course reaching **5,452 students** across the globe.  
- ✅ Feedback quality judged by **over 50 teaching assistants** and **LLM-as-a-judge analysis**.  
- ✅ The trained SLM closed the gap to GPT-4.1 from **80% → 10%** on correctness and helpfulness.  
- ✅ Supports hybrid strategies: SLMs for scalable local deployment, LLMs for more detailed diagnostic feedback.

<div align="center" style="margin:1.5em 0;">
  <img src="ta_feedback_overlay_criteria_by_exercise.pdf" alt="Feedback results comparison" width="500">
  <p><em>Trained 3B SLMs approach GPT-4.1 on correctness and helpfulness, while being locally deployable.</em></p>
</div>

---

## Impact

This study is the **first deployment of trained SLMs for diagnostic programming feedback in a global MOOC**.  
It demonstrates that small, open-source models can provide **timely, constructive, and scalable feedback**, reducing dependence on proprietary LLMs and enabling privacy-preserving educational tools.

---

## Contributors

We thank all the **teaching assistants** who evaluated feedback quality and the many **section leaders** who made Code In Place possible.  
The list of full contributors can be found on the [contributors page](https://koutche.me/cip25-aiep/contributors)

**Authors**:  
- [Charles Koutcheme](https://koutche.me/) (Aalto University)  
- [Juliette Woodrow](https://juliettewoodrow.github.io/) (Stanford University)  
- [Chris Piech](https://stanford.edu/~cpiech/bio/index.html) (Stanford University)  
