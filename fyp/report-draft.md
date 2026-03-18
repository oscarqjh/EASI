# Advancing Spatial and Embodied Intelligence in Multimodal Foundation Models: Evaluation, Improvement, and Interactive Reasoning

**Final Year Project Report**

---

## Abstract

This report documents the research conducted during my attachment at SenseTime Research, spanning January to March 2026. The work centers on three interconnected threads: (1) building comprehensive evaluation infrastructure for spatial intelligence (SI) in multimodal large language models (MLLMs), (2) contributing to efforts to improve spatial capabilities through data-centric training, and (3) extending evaluation from static benchmarks to interactive embodied reasoning. I contributed to two publications—*EASI: Holistic Evaluation of Multimodal LLMs on Spatial Intelligence* and *Scaling Spatial Intelligence with Multimodal Foundation Models*—primarily through benchmark integration, evaluation execution, and model debugging within the lmms-eval framework. Additionally, I independently designed and built the EASI embodied evaluation library, a unified toolkit for running simulator-based embodied AI benchmarks with LLM-powered agents.

---

## 1. Introduction

### 1.1 Motivation

Spatial intelligence—the ability to perceive, reason about, and interact with the three-dimensional physical world—is a cornerstone of artificial general intelligence. While multimodal foundation models have achieved remarkable progress on general vision-language tasks, they continue to exhibit notable limitations in spatial understanding [Cai et al., 2025a]. This gap between general multimodal competence and spatial reasoning capability motivates systematic study along three dimensions:

1. **Evaluation**: How do we comprehensively measure spatial intelligence across diverse tasks and models?
2. **Improvement**: Can data-centric approaches—scaling and curating spatial training data—close the gap?
3. **Embodied reasoning**: Do static benchmark gains transfer to interactive, embodied settings where agents must perceive, plan, and act in simulated environments?

These three questions form the narrative arc of this report and correspond to three phases of work conducted during my attachment.

### 1.2 Report Structure

- **Section 2** provides a literature review of spatial intelligence evaluation, multimodal model training, and embodied AI benchmarks.
- **Section 3** describes the EASI holistic evaluation framework (Paper 1) and my contributions to it.
- **Section 4** covers the SenseNova-SI data scaling work (Paper 2) and my role in benchmark integration and model evaluation.
- **Section 5** presents the EASI embodied evaluation library, which I designed and built independently to extend spatial intelligence evaluation into interactive settings.
- **Section 6** discusses findings, connections between the three phases, and future directions.
- **Section 7** concludes.

---

## 2. Literature Review

### 2.1 Spatial Intelligence in Multimodal Models

Spatial intelligence encompasses a range of capabilities that ground language understanding in the physical world. The EASI taxonomy [Cai et al., 2025a] decomposes SI into five fundamental capabilities:

- **Metric Measurement (MM)**: Understanding physical scale, estimating distances and sizes.
- **Spatial Relations (SR)**: Reasoning about relative positions in 3D (front-back, left-right, near-far).
- **Mental Reconstruction (MR)**: Inferring 3D object structure from limited 2D observations.
- **Perspective-Taking (PT)**: Reasoning about viewpoint changes—view correspondence, camera motion, allocentric transformations.
- **Comprehensive Reasoning (CR)**: Multi-step spatial reasoning coordinating multiple capabilities.

Recent studies reveal that even the most capable models (GPT-5, Gemini-2.5-Pro) fall significantly short of human performance on SI tasks, with gaps exceeding 30 percentage points on challenging subtasks like perspective-taking and metric measurement [Cai et al., 2025a].

### 2.2 Benchmarks for Spatial Reasoning

A growing ecosystem of benchmarks targets spatial capabilities:

- **VSI-Bench** [Li et al., 2024]: Video-based spatial intelligence with object counting, distance estimation, route planning, and spatial relations, evaluated on egocentric video.
- **MMSI** [Wang et al., 2024]: Multi-image spatial intelligence covering positional relationships (camera-camera, object-object, region-region), attributes, and motion.
- **MindCube** [Zhang et al., 2024]: Mental rotation tasks requiring perspective-taking (rotation, among, around subtasks).
- **ViewSpatial** [Lin et al., 2024]: View-dependent spatial reasoning across multiple viewpoints.
- **SITE** [Sun et al., 2024]: Spatial intelligence tasks in diverse environments covering 3D understanding, spatial relationship reasoning, motion prediction, and navigation.
- **OmniSpatial** [Chen et al., 2024]: Broad spatial reasoning across dynamic reasoning, spatial interaction, complex logic, and perspective-taking.
- **SparBench** [various]: Spatial arrangement reasoning.
- **3DSR-Bench** [various]: 3D spatial reasoning from images.
- **OSI-Bench** [various]: Object-centric spatial intelligence.

However, each benchmark covers only a subset of spatial capabilities. The EASI framework addresses this fragmentation by unifying them under a common taxonomy and evaluation protocol.

### 2.3 Data-Centric Approaches to Spatial Intelligence

Efforts to improve spatial reasoning in MLLMs follow two main strategies:

1. **Architecture-based**: Incorporating 3D expert encoders (e.g., Spatial-MLLM with VGGT, VLM-3R with geometry tokens, 3DThinker with 3D feature alignment).
2. **Data-centric**: Curating spatial training data for supervised fine-tuning. SpatialVLM [Chen et al., 2024] pioneered this with 2B synthetic VQA samples. Subsequent works include SpatialLadder (26K samples, progressive training), VST (4.1M SFT + 135K RL), and Cambrian-S (590K spatial video data).

The SenseNova-SI project [Cai et al., 2025b] takes the data-centric approach to scale, curating 8M diverse samples guided by the EASI taxonomy.

### 2.4 Embodied AI Benchmarks

Embodied AI evaluates agents that perceive, reason, and act in simulated 3D environments:

- **EmbodiedBench** [Yang et al., 2024]: Multi-domain embodied evaluation spanning household tasks (EB-Alfred), navigation (EB-Navigation), habitat tasks (EB-Habitat), and robotic manipulation (EB-Manipulation).
- **HAZARD** [Zhou et al., 2024]: Emergency evacuation under fire, flood, and wind scenarios—tests situational goal reasoning.
- **VLN-CE** [Krantz et al., 2020]: Vision-and-language navigation in continuous environments using Habitat simulator.
- **ManipulaTHOR** [Ehsani et al., 2021]: Arm-based point navigation requiring precise spatial manipulation.

These benchmarks require running actual simulators (AI2-THOR, Habitat-Sim, CoppeliaSim, ThreeDWorld), each with different Python dependencies and GPU requirements, making unified evaluation challenging.

---

## 3. EASI: Holistic Evaluation of Spatial Intelligence (Paper 1)

### 3.1 Overview

The first paper, *"Holistic Evaluation of Multimodal LLMs on Spatial Intelligence"* [Cai et al., 2025a], introduces EASI as a comprehensive framework for evaluating MLLMs on spatial reasoning. The study evaluates leading models—GPT-5, Gemini-2.5-Pro, Grok-4, Seed-1.6, Qwen2.5-VL, and InternVL3—across eight key benchmarks at a cost exceeding ten billion tokens.

### 3.2 Key Findings (Team Results)

The team's evaluation revealed several important findings:

1. **GPT-5 demonstrates unprecedented SI strength** but still falls significantly short of human performance (gaps of 24–55 percentage points across benchmarks).
2. **SI tasks expose greater model deficiency** than non-SI tasks—the gap between models and humans is substantially larger on spatial tasks.
3. **Proprietary models lack decisive advantage** on the hardest SI tasks—open-source models are competitive on challenging subtasks like perspective-taking.
4. **Perspective-taking and metric measurement** remain the weakest capabilities across all models.
5. **Qualitative analysis** reveals that advanced models fail on scenarios intuitive for humans, indicating fundamental limitations in physical world understanding.

### 3.3 My Contributions

My contributions to this paper centered on the **evaluation infrastructure and benchmark integration** within the lmms-eval framework:

#### 3.3.1 Benchmark Integration

I integrated multiple spatial intelligence benchmarks into the lmms-eval evaluation framework, enabling standardized, reproducible evaluation across models:

- **VSI-Bench (debiased and pruned)**: Integrated the debiased variant that removes annotation shortcuts, plus a pruned version for efficient evaluation.
- **VSI-Bench multi-image variant**: Extended the benchmark to support multi-image input format, crucial for models that process frame sequences differently.
- **ViewSpatial Bench**: Integrated the view-dependent spatial reasoning benchmark.
- **SITE Bench**: Added the spatial intelligence task evaluation benchmark.
- **SparBench**: Integrated the spatial arrangement reasoning benchmark.
- **MMSI-Video Bench**: Added the video variant of multi-image spatial intelligence.
- **OSI-Bench**: Integrated the object-centric spatial intelligence benchmark.
- **3DSR-Bench**: Added 3D spatial reasoning evaluation.
- **MindCube-Tiny (bugfix)**: Fixed evaluation issues in the compact MindCube variant.

#### 3.3.2 Model Integration and Debugging

I also contributed to model support within lmms-eval:

- **Cambrian-S model**: Added support for the spatial intelligence model family (3B and 7B variants).
- **Bagel model**: Integrated the unified understanding-and-generation model.
- **Bug fixes**: Fixed the Qwen3-VL nframes bug (incorrect frame sampling), the InternVL image token bug for interleaved multi-image format, and various issues for LLaVA-OV-1.5, InternVL3, and Cambrian-S models.

#### 3.3.3 Evaluation Execution

Each benchmark integration involved not just code implementation but also running the full evaluation pipeline across multiple models, verifying results against published baselines, and debugging discrepancies. The VSI-Bench debiased variant, for instance, required careful validation that the debiasing procedure correctly removed annotation shortcuts without altering the benchmark's intended difficulty.

### 3.4 Significance

My infrastructure work enabled the team to evaluate a comprehensive set of benchmarks under a unified framework, which was essential for the paper's core contribution: a holistic, apples-to-apples comparison across models and spatial capabilities. The lmms-eval integrations I built are publicly available and continue to be used by the research community for spatial intelligence evaluation.

---

## 4. Scaling Spatial Intelligence (Paper 2)

### 4.1 Overview

The second paper, *"Scaling Spatial Intelligence with Multimodal Foundation Models"* [Cai et al., 2025b], investigates whether data-centric scaling can improve spatial intelligence in MLLMs. The team built the SenseNova-SI family of models by fine-tuning established foundations (Qwen3-VL, InternVL3, Bagel) on SenseNova-SI-8M: eight million diverse spatial QA samples organized under the EASI taxonomy.

### 4.2 Approach (Team Work)

The data curation strategy, training methodology, and core analysis were conducted by the team:

- **Data taxonomy**: Following the EASI five-capability decomposition (MM, SR, PT, MR, CR), the team reorganized 4M open-source samples and generated 4.5M additional samples to address gaps, particularly in perspective-taking.
- **Data sources**: General QA (0.6M), community spatial datasets (3.3M from Open3D-VQA, CLEVR-series, MultiSpa, MindCube, VSI-590K, etc.), and further scaling (4.5M) using 3D datasets (ScanNet, ScanNet++, Matterport3D, Ego-Exo4D, etc.).
- **Training**: Supervised fine-tuning on three base models without architectural changes—purely data-centric.
- **Key findings**:
  - **Scaling laws**: Distinct scaling behaviors across spatial capabilities and model sizes; saturation trends suggest future advances may require paradigm shifts.
  - **Emergent generalization**: Models trained on one set of spatial tasks exhibit transfer to seemingly unrelated tasks and extrapolation to longer spatial contexts.
  - **Robustness**: Controlled experiments validate genuine spatial capability acquisition, not memorization or shortcut exploitation.
  - **Spatial CoT limitations**: Text-based chain-of-thought prompting does not reliably improve spatial reasoning beyond QA-style data scaling.
  - **Downstream validation**: SenseNova-SI applied to EmbodiedBench robotic manipulation without fine-tuning shows performance improvements.

### 4.3 My Contributions

My contributions to this paper were focused on the **evaluation and validation** side:

#### 4.3.1 Benchmark Evaluation Infrastructure

The SenseNova-SI models needed to be evaluated on the same spatial benchmarks I had previously integrated into lmms-eval (Section 3.3.1). My benchmark integrations—VSI-Bench, MMSI, MindCube-Tiny, ViewSpatial, SITE, and others—formed the evaluation backbone for this paper. Specifically, the lmms-eval framework I helped build was used to produce the results reported in Table 1 (main results) and Tables 9–13 (detailed per-benchmark breakdowns).

#### 4.3.2 Model Support

The Cambrian-S and Bagel model integrations I built in lmms-eval were used as baselines in this paper. The SenseNova-SI Bagel-7B-MoT variant builds directly on the Bagel model I integrated.

#### 4.3.3 Evaluation Execution

I ran evaluations of the SenseNova-SI models across the integrated benchmarks, verifying that results were consistent and identifying edge cases in model inference (e.g., frame sampling, image token formatting).

### 4.4 Connection to Paper 1

The two papers form a natural pair: Paper 1 establishes *where models stand* on spatial intelligence and provides the evaluation methodology; Paper 2 asks *how to improve* using the insights and infrastructure from Paper 1. My work bridged these two efforts by providing the shared evaluation infrastructure.

---

## 5. EASI Embodied Evaluation Library

### 5.1 Motivation

The first two papers evaluate spatial intelligence through static benchmarks—models receive images/videos and answer questions. But spatial intelligence ultimately matters because it enables agents to act in the physical world. A natural question emerges: **do spatial reasoning gains translate to better performance in interactive, embodied settings?**

To investigate this, I independently designed and built the **EASI embodied evaluation library** (`easi`), a unified toolkit for running simulator-based embodied AI benchmarks with LLM-powered agents. This extends the EASI evaluation philosophy from static Q&A to interactive agent evaluation.

### 5.2 Design Goals

1. **Unified interface**: A single CLI (`easi start`) to run any embodied benchmark with any LLM backend.
2. **Simulator isolation**: Each simulator (AI2-THOR, Habitat-Sim, CoppeliaSim, ThreeDWorld) runs in its own subprocess with its own conda environment, handling Python version conflicts (e.g., AI2-THOR v2.1 requires Python 3.8 while the host runs Python 3.10+).
3. **Multi-split tasks**: Benchmarks like EmbodiedBench have multiple evaluation splits (e.g., EB-Alfred has 6 splits testing different capabilities). Each split is auto-discovered and registered as a separate task.
4. **Pluggable agents**: Support for dummy agents (testing), ReAct agents with multi-action buffering, and arbitrary LLM backends (OpenAI, Anthropic, Gemini, vLLM).
5. **Reproducibility**: Full logging of trajectories, observations, and metrics for every episode.

### 5.3 Architecture

The library follows a layered architecture:

```
┌─────────────────────────────────────────────────────┐
│                    CLI (easi)                         │
│  start | task list | env install | sim test | ...     │
├─────────────────────────────────────────────────────┤
│              Evaluation Runner                        │
│  Sequential / Parallel (thread-pool)                  │
│  Episode filtering, resume, metrics aggregation       │
├──────────────┬──────────────────────────────────────┤
│   Agent      │         Task                           │
│  ReAct agent │  BaseTask + YAML config                │
│  + Prompt    │  Per-episode evaluation                 │
│    Builder   │  Auto-discovered from registry          │
├──────────────┼──────────────────────────────────────┤
│         Simulator (subprocess-isolated)               │
│  Bridge script ←──filesystem IPC──→ Parent process    │
│  Own conda env, own Python version                    │
└─────────────────────────────────────────────────────┘
```

Key architectural decisions:

- **Subprocess isolation via filesystem IPC**: The bridge script in each simulator's conda environment communicates with the parent process through atomic JSON file reads/writes. This avoids the complexity of socket-based IPC while maintaining reliability across Python version boundaries.
- **Auto-discovery**: Simulators are discovered via `manifest.yaml` files; tasks via `*.yaml` glob patterns. Adding a new benchmark requires only creating a task directory with a YAML config and a Python class—no registry modification needed.
- **YAML template inheritance**: Multi-split tasks use an `extends: _base.yaml` mechanism to share common configuration while varying split-specific parameters (dataset subset, episode IDs, action space).

### 5.4 Integrated Benchmarks

I integrated the following embodied benchmarks, each requiring understanding of the underlying simulator, environment setup, episode format, action space, and evaluation metrics:

| Benchmark | Simulator | Splits | Capability Tested |
|-----------|-----------|--------|-------------------|
| EB-Alfred | AI2-THOR v2.1 | 6 (base, spatial, visual, commonsense, complex, long-horizon) | Household task completion |
| EB-Navigation | AI2-THOR v5.0 | 5 | Indoor point-goal navigation |
| EB-Habitat | Habitat-Sim | 4 | Scene understanding + navigation |
| EB-Manipulation | CoppeliaSim | 4 | Robotic manipulation |
| HAZARD | ThreeDWorld | 3 (fire, flood, wind) | Situational goal reasoning under emergencies |
| ManipulaTHOR | AI2-THOR | 2 (seen, unseen) | Arm-based point navigation |
| AI2-THOR Rearrangement | AI2-THOR | 5 | Object rearrangement |
| VLN-CE R2R | Habitat-Sim | 2 (val seen, unseen) | Vision-language navigation |
| VLN-CE RxR | Habitat-Sim | 5 (val seen/unseen, en/hi/te) | Multilingual vision-language navigation |
| LHPR-VLN | Habitat-Sim | Multiple (val/test) | Multi-subtask navigation |

Each integration involved:
1. Setting up the simulator environment (conda env with specific Python version and dependencies)
2. Writing the bridge script for subprocess IPC
3. Implementing the task class with episode loading, reset configuration, and metric computation
4. Writing the prompt builder for the ReAct agent (translating observations and actions to/from natural language)
5. Running verification episodes to confirm correct behavior

### 5.5 The ReAct Agent

The library includes a ReAct (Reasoning + Acting) agent that interfaces with any LLM backend to control embodied agents:

1. **Observation → Prompt**: The PromptBuilder converts simulator observations (images, object lists, task descriptions) into structured prompts.
2. **LLM Reasoning**: The model receives the prompt and generates a thought + action (or multi-action plan).
3. **Action Execution**: The agent parses the LLM response, validates the action against the task's action space, and executes it in the simulator.
4. **Multi-action buffering**: When the LLM proposes a plan of multiple actions, the agent executes them one at a time, clearing the buffer on failure to allow replanning.

### 5.6 Evaluation Infrastructure

- **Parallel evaluation**: Thread-pool parallelism with configurable worker count, supporting multiple vLLM instances with GPU allocation control.
- **Episode filtering**: Flexible `--episodes` flag supporting IDs (`2,5,7`), ranges (`10:20`), and limits (`:5`).
- **Resume**: Interrupted runs can be resumed from the last completed episode.
- **Structured output**: Per-episode trajectories (JSONL), observation images, result metrics, and run-level summaries.
- **Trajectory analysis**: `easi analyze trajectory` generates step-by-step videos from logged observations.

### 5.7 Connecting Static and Embodied Evaluation

The EASI embodied library enables direct investigation of whether spatial intelligence improvements (from Paper 2's data scaling) transfer to embodied settings. The SenseNova-SI paper already demonstrated this connection in a preliminary fashion: applying SenseNova-SI to EmbodiedBench manipulation tasks without fine-tuning showed performance improvements, suggesting that spatial data scaling benefits extend beyond static benchmarks.

The embodied library makes it straightforward to run such transfer experiments: the same `easi start` command works with any model backend, allowing researchers to swap in a spatially-trained model and measure impact across diverse embodied tasks.

---

## 6. Discussion

### 6.1 A Coherent Research Arc

The three phases of work form a natural progression:

1. **Evaluate** (Paper 1): Establish where current models stand on spatial intelligence through comprehensive benchmarking. Key finding: all models, including GPT-5, fall significantly short of human spatial intelligence.

2. **Improve** (Paper 2): Use the evaluation insights to guide data-centric improvement. The EASI taxonomy reveals which capabilities are weakest (perspective-taking, metric measurement), guiding targeted data scaling. Result: SenseNova-SI achieves state-of-the-art on spatial benchmarks.

3. **Extend** (EASI Library): Push beyond static evaluation to interactive embodied reasoning. Question: do static benchmark gains translate to better agent behavior in simulated environments? The library provides the infrastructure to systematically investigate this.

### 6.2 Insights on the Gap Between Static and Embodied Intelligence

Static spatial intelligence benchmarks test *perception and reasoning*—the model observes and answers. Embodied benchmarks additionally require *planning and acting*—the model must decompose goals, execute multi-step plans, and recover from failures. The EASI embodied library reveals additional capability dimensions beyond those captured by static benchmarks:

- **Long-horizon planning**: EB-Alfred complex and long-horizon splits require 20+ step plans.
- **Situational reasoning**: HAZARD scenarios require inferring implicit goals (e.g., "evacuate valuable objects during a fire" is not explicitly stated).
- **Continuous navigation from language**: VLN-CE requires translating free-form navigation instructions into continuous movement.
- **Spatial manipulation**: EB-Manipulation tests precise 3D positioning and gripper control.

### 6.3 Limitations and Future Work

- **Evaluation scale**: Running embodied evaluations is orders of magnitude more expensive than static benchmarks (each episode requires multiple simulator steps and LLM calls). Efficient evaluation strategies are needed.
- **Transfer study**: While SenseNova-SI showed preliminary improvements on EmbodiedBench manipulation, a comprehensive study across all integrated benchmarks remains future work.
- **Agent architecture**: The ReAct agent is a baseline; more sophisticated agent architectures (e.g., with explicit spatial representations, memory, and hierarchical planning) could better leverage spatial intelligence.
- **Beyond English**: The VLN-CE RxR benchmark includes Hindi and Telugu instructions, enabling multilingual embodied evaluation—an underexplored area.

---

## 7. Conclusion

This report presents a cohesive body of work advancing the evaluation and improvement of spatial intelligence in multimodal foundation models. Through contributions to two publications and the independent development of the EASI embodied evaluation library, I engaged with the full lifecycle of spatial intelligence research: measuring current capabilities, improving them through data scaling, and extending evaluation to interactive embodied settings.

The evaluation infrastructure I built within lmms-eval—spanning 9+ spatial benchmarks and multiple model integrations—enabled the holistic evaluation reported in Paper 1 and the validation of data scaling results in Paper 2. The EASI embodied library I independently designed and built provides the research community with a unified toolkit for embodied AI evaluation, integrating 10 benchmarks across 5 simulators with LLM-powered agent support.

Together, these contributions advance our understanding of where spatial intelligence stands, how to improve it, and how to evaluate whether improvements matter in the real-world-adjacent settings that ultimately define spatial intelligence's utility.

---

## References

[Cai et al., 2025a] Zhongang Cai, Yubo Wang, Qingping Sun, et al. "Holistic Evaluation of Multimodal LLMs on Spatial Intelligence." arXiv:2508.13142, 2025.

[Cai et al., 2025b] Zhongang Cai, Ruisi Wang, Chenyang Gu, et al. "Scaling Spatial Intelligence with Multimodal Foundation Models." arXiv:2511.13719, 2025.

[Chen et al., 2024] Boyuan Chen et al. "SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities." CVPR, 2024.

[Ehsani et al., 2021] Kiana Ehsani et al. "ManipulaTHOR: A Framework for Visual Object Manipulation." CVPR, 2021.

[Krantz et al., 2020] Jacob Krantz et al. "Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments." ECCV, 2020.

[Li et al., 2024] Various. "VSI-Bench: Video Spatial Intelligence Benchmark."

[Yang et al., 2024] Various. "EmbodiedBench: Comprehensive Benchmarking of Multi-modal LLMs as Embodied Agents."

[Zhou et al., 2024] Various. "HAZARD: Emergency Evacuation Benchmark for Embodied AI."

---

## Appendix A: Summary of Personal Contributions

### A.1 Benchmark Integration (lmms-eval)

| Week | Benchmark/Task | Type |
|------|---------------|------|
| 5/1–9/1 | VSI-Bench debiased/pruned, VSI-Bench multi-image, ViewSpatial, SITE | New benchmark |
| 12/1–16/1 | VSI-Bench multi-image (continued), lmms-eval as EASI backend | Integration |
| 19/1–23/1 | SparBench | New benchmark |
| 26/1–30/1 | MMSI-Video | New benchmark |
| 2/2–6/2 | OSI-Bench | New benchmark |
| 9/2–13/2 | 3DSR-Bench, MindCube-Tiny (bugfix) | New benchmark + fix |

### A.2 Model Integration (lmms-eval)

| Week | Model/Fix | Type |
|------|-----------|------|
| 5/1–9/1 | Cambrian-S model | New model |
| 19/1–23/1 | Bagel model, Qwen3-VL nframes fix, InternVL interleave fix | New model + bugfixes |
| 9/2–13/2 | LLaVA-OV-1.5, InternVL3, Cambrian-S bugfixes | Bugfixes |

### A.3 EmbodiedBench Environment Verification

| Week | Task |
|------|------|
| 26/1–30/1 | Verified EB-Manipulation, EB-Alfred, EB-Habitat, EB-Navigation environments |
| 2/2–6/2 | Verified Alfred and Habitat on server |

### A.4 EASI Embodied Library (Independent Work)

| Period | Work |
|--------|------|
| 16/2–13/3 | Full library design and implementation |
| | Core architecture: CLI, subprocess isolation, filesystem IPC, auto-discovery |
| | EmbodiedBench: 19 task splits across 4 domains |
| | HAZARD: 3 emergency scenarios |
| | ManipulaTHOR: 2 splits (seen/unseen) |
| | AI2-THOR Rearrangement: 5 evaluation splits |
| | VLN-CE R2R: 2 splits (val seen/unseen) |
| | VLN-CE RxR: 5 splits (multilingual) |
| | LHPR-VLN: Multi-subtask navigation |
| | ReAct agent with multi-action buffering |
| | Parallel evaluation with vLLM support |
| | Trajectory analysis and video generation |
| | 540+ tests, full offline test suite |

---

## Appendix B: Distinction of Contributions

To clearly distinguish between team contributions and my individual work:

### Team Contributions (I did NOT do these)
- EASI spatial taxonomy design and conceptualization
- Paper writing and narrative framing
- SenseNova-SI-8M dataset curation and synthesis (the 8M spatial data samples)
- Model training and SFT experiments
- Data scaling analysis, emergent capability analysis, overfitting studies
- Spatial chain-of-thought experiments
- Qualitative evaluation and failure case analysis
- OmniSpatial benchmark integration (was not part of my scope)
- Core EASI evaluation protocol design

### My Contributions
- **lmms-eval benchmark integrations**: VSI-Bench (debiased, pruned, multi-image), ViewSpatial, SITE, SparBench, MMSI-Video, OSI-Bench, 3DSR-Bench, MindCube-Tiny bugfix — 9 benchmark integrations total
- **lmms-eval model integrations**: Cambrian-S, Bagel — 2 model integrations
- **Model debugging**: Qwen3-VL nframes, InternVL interleave tokens, LLaVA-OV-1.5, InternVL3, Cambrian-S — 5+ bugfixes
- **Evaluation execution**: Running evaluations across models and benchmarks, verifying results
- **EmbodiedBench environment verification**: Setting up and verifying all 4 EmbodiedBench domains on server
- **EASI embodied evaluation library**: Entire library independently designed and implemented (architecture, CLI, 10 benchmarks, 5 simulators, agents, evaluation pipeline, 540+ tests)
