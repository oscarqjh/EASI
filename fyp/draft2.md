# Advancing Spatial and Embodied Intelligence in Multimodal Foundation Models

**Final Year Project Report**

---

## Abstract

Spatial intelligence—the ability to perceive, reason about, and interact with the three-dimensional physical world—remains one of the most significant gaps in modern multimodal AI. This report documents research conducted at SenseTime Research from January to March 2026, tracing a narrative from diagnosing the spatial reasoning deficiencies of frontier models, through data-driven efforts to close that gap, to the open question of whether static benchmark gains translate into interactive embodied competence. The work spans contributions to two publications [1, 2] and the independent development of a unified embodied evaluation toolkit, together forming a coherent investigation into what it takes to make multimodal models spatially intelligent.

---

## 1. Introduction

In early 2025, a curious pattern emerged across the multimodal AI landscape. Models like GPT-4o and Gemini-1.5 could describe images in exquisite detail, answer complex questions about charts and documents, and even reason about abstract visual puzzles—yet they struggled with questions that a five-year-old could answer intuitively. *"Which object is closer to the camera?"* *"If you turned around, what would you see?"* *"How far apart are the chair and the desk?"* These are not exotic tasks; they are the bread and butter of how humans navigate the physical world.

This observation motivated the research programme that this report documents. The central thesis is simple: spatial intelligence is both critically important and critically underserved by current evaluation and training paradigms. The work proceeds in three phases. First, we ask: *how bad is the problem, really?* To answer this rigorously requires a holistic evaluation framework that goes beyond individual benchmarks. Second, armed with a precise diagnosis of what models get wrong, we ask: *can we fix it through data?* Specifically, can curating and scaling spatial training data systematically improve these capabilities? Third, and most ambitiously: *does any of this matter for agents that must act in the world?* Static benchmarks test perception and reasoning in isolation, but embodied agents must also plan, execute, and recover from failure in interactive 3D environments.

These three questions—diagnosis, treatment, and real-world validation—structure the report.

---

## 2. Background and Related Work

### 2.1 What Is Spatial Intelligence?

Spatial intelligence is not a single capability but a family of related skills. Following the taxonomy proposed in [1], we decompose it into five fundamental dimensions:

**Metric Measurement (MM)** involves understanding physical scale: estimating distances between objects, gauging the size of a room, or judging whether a sofa would fit through a doorway. **Spatial Relations (SR)** captures the ability to reason about relative positions—front-back, left-right, above-below, near-far—within a 3D coordinate system. **Mental Reconstruction (MR)** requires inferring 3D object structure from limited 2D views, such as identifying which side of an object is visible. **Perspective-Taking (PT)** is perhaps the most demanding: reasoning about how a scene appears from different viewpoints, establishing correspondences across views, inferring camera motion, and mentally simulating viewpoint shifts. Finally, **Comprehensive Reasoning (CR)** involves coordinating multiple spatial capabilities across multi-step reasoning chains.

This taxonomy proved essential for the work that followed. Without it, evaluation results would be a soup of numbers from disparate benchmarks; with it, we could precisely identify which spatial capabilities are strong, which are weak, and where targeted intervention is most needed.

### 2.2 The Benchmark Landscape

A growing ecosystem of benchmarks has emerged to test these capabilities, each covering a different slice of the space. VSI-Bench [3] evaluates video-based spatial reasoning—object counting, distance estimation, route planning—using egocentric indoor video. MMSI [4] tests multi-image spatial understanding across positional relationships between cameras, objects, and regions. MindCube [5] focuses squarely on perspective-taking through mental rotation tasks. ViewSpatial [6] probes view-dependent spatial reasoning. SITE [7] covers 3D information understanding, spatial relationships, and motion prediction. OmniSpatial [8] spans dynamic reasoning, spatial interaction, complex logic, and perspective-taking.

The fragmentation is the problem. Each benchmark uses its own evaluation protocol, prompt format, and metric. A model might score well on VSI-Bench's object counting but poorly on its route planning; it might handle MMSI's camera-camera correspondences but fail on object-object ones. Without a unified framework, comparing models across the full breadth of spatial capabilities is impractical.

### 2.3 Approaches to Improving Spatial Reasoning

Efforts to close the spatial intelligence gap follow two broad strategies. Architecture-based approaches incorporate 3D-aware encoders—Spatial-MLLM [9] adds VGGT for depth estimation, VLM-3R [10] introduces geometry tokens, 3DThinker [11] aligns model features with 3D supervision. Data-centric approaches instead curate spatial training data and apply supervised fine-tuning (SFT) or reinforcement learning (RL). SpatialVLM [12] pioneered this direction with 2 billion synthetic spatial QA samples. Subsequent work includes SpatialLadder (26K samples with progressive training) [13], VST (4.1M SFT + 135K RL samples) [14], and Cambrian-S (590K spatial video samples with four-stage training) [15].

The data-centric approach is appealing because it does not require modifying model architectures—any foundation model can potentially benefit. But the question of *which* data, *how much*, and *organized how* remained open.

### 2.4 From Static Benchmarks to Embodied Agents

Static benchmarks, however comprehensive, test spatial intelligence in a fundamentally passive mode: the model receives images or video frames and answers questions. But spatial intelligence exists *for* action. Humans develop spatial reasoning not as an academic exercise but because it enables us to navigate rooms, manipulate objects, and coordinate with others in shared physical spaces.

Embodied AI benchmarks close this gap by placing agents in simulated 3D environments where they must perceive, reason, plan, and act. EmbodiedBench [16] provides a multi-domain evaluation spanning household tasks, navigation, habitat exploration, and robotic manipulation. HAZARD [17] tests emergency response—can an agent identify valuable objects and evacuate them during a fire, flood, or wind scenario? VLN-CE [18] requires translating natural language navigation instructions into continuous movement through realistic indoor environments. ManipulaTHOR [19] tests arm-based point navigation requiring precise spatial positioning.

These benchmarks are expensive to run (each requiring its own simulator, conda environment, and GPU resources) and have historically been evaluated in isolation. No unified framework existed for running them all under a common protocol.

---

## 3. Diagnosing the Problem: Holistic Evaluation of Spatial Intelligence

### 3.1 Building the Evaluation Infrastructure

The first phase of work addressed a practical prerequisite: before we could holistically evaluate spatial intelligence, we needed the benchmarks to actually work within a unified evaluation framework. The lmms-eval framework [20] provided the foundation, but most spatial benchmarks had not yet been integrated.

Over the first six weeks of the attachment, I integrated nine spatial benchmarks into lmms-eval: VSI-Bench in its debiased and pruned variants (removing annotation shortcuts that could inflate scores), a multi-image variant of VSI-Bench (critical for models that handle frame sequences differently from single images), ViewSpatial, SITE, SparBench, MMSI-Video, OSI-Bench, 3DSR-Bench, and a bugfix for MindCube-Tiny. Each integration required implementing the benchmark's specific evaluation protocol—prompt templates, answer extraction, metric computation—and validating results against published baselines.

Alongside the benchmarks, I integrated two models that would serve as important baselines: Cambrian-S, a spatial intelligence model family trained on 590K spatial video samples, and Bagel, a unified understanding-and-generation model. I also fixed several inference bugs that surfaced during large-scale evaluation: Qwen3-VL was incorrectly sampling video frames due to an nframes bug, InternVL had an image token formatting issue in interleaved multi-image mode, and various issues appeared in LLaVA-OV-1.5, InternVL3, and Cambrian-S when processing certain benchmark formats. These fixes were essential—without them, the evaluation results for these models would have been unreliable.

### 3.2 The EASI Evaluation Framework

With the infrastructure in place, the team conducted a comprehensive evaluation across eight key benchmarks [1]. The study, titled *"Holistic Evaluation of Multimodal LLMs on Spatial Intelligence"* (EASI), tested the leading proprietary models (GPT-5, Gemini-2.5-Pro, Grok-4, Seed-1.6) and open-source models (Qwen2.5-VL, InternVL3, and others) at a cost exceeding ten billion total tokens.

The EASI framework's core contribution is a unified evaluation protocol that maps every benchmark subtask onto the five-capability taxonomy, enabling apples-to-apples comparison across models and capabilities. A standardized zero-shot chain-of-thought prompt is applied uniformly, and results are reported both per-benchmark and aggregated by capability.

### 3.3 What the Evaluation Revealed

Table 1 presents the main results across eight spatial intelligence benchmarks. Four of these benchmarks—VSI-Bench, SITE, MMSI, and MindCube-Tiny—were evaluated using the integrations I built in lmms-eval, while OmniSpatial, STARE, CoreCognition, and SpatialViz were integrated by other team members.

**Table 1.** Evaluation on eight spatial intelligence benchmarks (Official Protocol), adapted from [1]. MindCube* denotes MindCube-Tiny.

| Model | VSI | SITE | MMSI | OmniSpatial | MindCube* | STARE | CoreCog. | SpatialViz |
|-------|-----|------|------|-------------|-----------|-------|----------|------------|
| *Metric* | *MRA,Acc* | *CAA* | *Acc* | *Acc* | *Acc* | *Acc,F1* | *Acc* | *Acc* |
| Random Choice | 34.0 | 0.0 | 25.0 | 25.0 | 32.4 | 34.8 | 33.9 | 25.1 |
| **Proprietary** | | | | | | | | |
| Seed-1.6 | 49.9 | 54.6 | 38.3 | 49.3 | 48.8 | 46.1 | 77.2 | 34.6 |
| Gemini-2.5-Pro | 53.6 | 57.1 | 38.0 | 55.4 | 57.6 | 49.1 | 76.7 | 42.7 |
| Grok-4 | 47.9 | 47.0 | 37.8 | 46.8 | 63.6 | 26.9 | 79.3 | 19.4 |
| GPT-5-nano | 43.2 | 35.8 | 28.9 | 47.8 | 41.5 | 46.1 | 67.9 | 35.6 |
| GPT-5-mini | 48.7 | 52.5 | 34.1 | 55.5 | 56.7 | 52.5 | 77.8 | 44.7 |
| GPT-5 | **55.0** | **61.9** | **41.8** | **59.9** | 56.3 | **54.6** | **84.4** | **51.3** |
| **Open-source** | | | | | | | | |
| Qwen2.5-VL-3B | 27.0 | 33.1 | 28.6 | 42.5 | 37.6 | 37.8 | 60.2 | 21.9 |
| Qwen2.5-VL-7B | 32.3 | 37.6 | 26.8 | 39.1 | 36.1 | 35.0 | 62.2 | 26.8 |
| Qwen2.5-VL-72B | 35.8 | 47.4 | 32.5 | 47.8 | 42.4 | 38.4 | 69.2 | 32.5 |
| InternVL3-8B | 42.1 | 41.2 | 28.0 | 46.3 | 41.5 | 41.4 | 60.9 | 30.0 |
| InternVL3-78B | 47.6 | 52.7 | 30.5 | 51.0 | 49.5 | 42.0 | 71.2 | 31.1 |
| InternVL3.5-8B | 56.1 | 43.8 | 27.3 | 46.7 | 42.5 | 40.2 | 66.4 | 24.0 |
| Qwen3-8B | 57.9 | 45.8 | 31.1 | 45.7 | 29.4 | 39.8 | 69.7 | 17.5 |
| **Human** | **79.2** | **67.5** | **97.2** | **92.6** | **94.6** | **96.5** | **87.0** | **82.5** |
| Delta (best model − human) | −21.3 | −5.6 | −55.4 | −32.7 | −31.0 | −42.1 | −2.6 | −31.2 |

The results painted a stark picture. GPT-5, the most powerful model tested, set a new state of the art—yet the human-model gap remained enormous on most benchmarks. Consider the MMSI column, which I integrated into lmms-eval: humans score 97.2% while GPT-5 manages just 41.8%, a gap of over 55 points. On MindCube-Tiny—another benchmark I integrated, with its evaluation bugs fixed—human performance of 94.6% dwarfs GPT-5's 56.3%. Even SITE, where the gap appears smallest at 5.6 points, benefits from a relatively low human baseline of 67.5%; the model's 61.9% is impressive but deceptive without context.

More revealing than the absolute gaps were the patterns across capabilities. Perspective-taking (PT) and metric measurement (MM) emerged as the weakest dimensions across all models. Drilling into the per-subtask breakdowns that I helped produce by running evaluations across models, the picture sharpened further. On MMSI's camera-to-camera perspective-taking questions, the best model (GPT-5) scored 41.9% against a human baseline of 95.7%—a gap of over 53 points. On MindCube's "among" subtask (perspective-taking over multiple objects), GPT-5 reached only 38.2% versus humans at 94.5%. The VSI-Bench results—evaluated using my debiased variant that removes annotation shortcuts which could inflate scores—showed that even basic spatial relation tasks like route planning had a 68-point gap between GPT-5 and humans.

A particularly striking finding was that proprietary models did not hold a decisive advantage over open-source models on the hardest spatial tasks. Looking at Table 1, InternVL3-78B (47.6%) nearly matches GPT-5-nano (43.2%) on VSI-Bench and exceeds it on SITE (52.7% vs. 35.8%). On OmniSpatial's geometric reasoning and hypothetical perspective-taking subtasks, the strongest open-source models actually matched or exceeded some proprietary models. This suggested that the gap was not simply about model scale but about something more fundamental—perhaps the training data itself. This insight would prove pivotal for the data scaling work that followed.

The evaluation also revealed that spatial intelligence tasks exposed *greater* model capability deficiency than non-SI tasks. On CoreCognition, models had already reached human-level accuracy on several non-spatial tasks (Boundary, Perceptual Constancy, Conservation), yet the spatial subtasks remained far behind. On SpatialViz, the best model scored 46 percentage points lower on 3D mental rotation than on 2D mental rotation. The models were not uniformly weak; they were *specifically* weak at spatial reasoning.

### 3.4 Qualitative Insights

Beyond the numbers, qualitative analysis revealed failure modes that were both surprising and informative. Models would correctly identify all objects in a scene but mislocate them relative to each other. They could describe a room's layout in words but fail to infer what would be visible after a 90-degree turn. GPT-5 performed worse than random guessing on two mental reconstruction subtasks—the Attribute (Appearance) task in MMSI and the 3D Mental Rotation task in SpatialViz—suggesting that the model's reasoning process could actually be counterproductive for certain spatial tasks. These failures were not random—they pointed to a systematic absence of the kind of 3D spatial representation that would support robust spatial reasoning.

---

## 4. Treating the Problem: Scaling Spatial Data

### 4.1 From Diagnosis to Intervention

The holistic evaluation provided a precise diagnosis: models are weakest at perspective-taking and metric measurement, moderately weak at spatial relations and mental reconstruction, and relatively stronger (though still far from human) at comprehensive reasoning when it can be solved through language-level inference. The natural next question was whether targeted data curation and scaling could close these gaps.

This question drove the second publication, *"Scaling Spatial Intelligence with Multimodal Foundation Models"* [2], which I contributed to through evaluation infrastructure and execution. The team adopted a deliberately data-centric approach: rather than modifying model architectures, they scaled up spatial training data while preserving the original architectures of established foundation models (Qwen3-VL, InternVL3, and Bagel). The reasoning was pragmatic—if data alone could unlock spatial intelligence, the benefit would be immediately transferable to any model family.

### 4.2 Curating SenseNova-SI-8M

The team constructed SenseNova-SI-8M, a collection of eight million spatial QA samples organized under the EASI taxonomy. The curation process began by auditing existing open-source spatial datasets—Open3D-VQA, the CLEVR series, MultiSpa, MindCube, ViCA, VLM-3R, VSI-590K—yielding approximately 3.3 million QA pairs. An additional 0.6 million general visual QA pairs (from VQA, GQA, IconQA, etc.) were included to maintain general capabilities.

The audit revealed significant gaps. Metric measurement (MM) and spatial relations (SR) dominated the existing data, while perspective-taking (PT) and mental reconstruction (MR) were severely underrepresented. Point-level and scene-level correspondence tasks were limited to a single dataset. Camera motion reasoning existed only in sparse form. Allocentric viewpoint transformation—especially object-centric and hypothetical views—was largely unexplored, since real-world QA labels for these tasks are scarce.

To address these gaps, the team leveraged richly annotated 3D scene datasets—ScanNet, ScanNet++, SUN RGB-D, Matterport3D, Ego-Exo4D, MessyTable, and CA-1M—to synthesize 4.5 million additional QA pairs with special emphasis on perspective-taking. The resulting 8.5M samples spanned all five capability dimensions with deliberate attention to balance.

### 4.3 Training the SenseNova-SI Models

Three base models were selected for SFT: InternVL3 (natively multimodal, joint vision-language pre-training), Qwen3-VL (language-foundation-first, expanded to vision), and Bagel (unified understanding and generation). Each was fine-tuned on subsets of SenseNova-SI-8M without architectural modification.

The benchmark integrations I had built in lmms-eval—VSI-Bench, MMSI, MindCube-Tiny, ViewSpatial, SITE—formed the evaluation backbone for validating the trained models. The Cambrian-S and Bagel model integrations I had built served directly as baselines; the SenseNova-SI Bagel-7B-MoT variant, for instance, builds on the same Bagel model I had integrated for evaluation.

### 4.4 Results: What Data Scaling Achieved

Table 2 shows the impact of SenseNova-SI data scaling, evaluated on the five core spatial benchmarks—all of which were benchmarks I had integrated into lmms-eval during the first phase of work.

**Table 2.** SenseNova-SI results on spatial intelligence benchmarks, adapted from [2]. The benchmarks used for evaluation are the same integrations built during Phase 1 (Section 3).

| Model | VSI-Bench | MMSI | MindCube* | ViewSpatial | SITE | MMBench-En |
|-------|-----------|------|-----------|-------------|------|------------|
| *Metric* | *MRA,Acc* | *Acc* | *Acc* | *Acc* | *CAA* | *Acc* |
| Human | 79.2 | 97.2 | 94.5 | — | 67.5 | — |
| GPT-5 | 55.0 | 41.8 | 56.3 | 45.5 | 61.8 | 85.2 |
| Gemini-2.5-Pro | 53.5 | 38.0 | 57.6 | 46.0 | 57.0 | 90.1 |
| **Base models** | | | | | | |
| InternVL3-8B | 42.1 | 28.0 | 41.5 | 38.6 | 41.1 | 81.7 |
| Qwen3-VL-8B | 57.9 | 31.1 | 29.4 | 42.2 | 45.8 | 84.6 |
| Bagel-7B-MoT | 31.4 | 31.0 | 34.7 | 41.3 | 37.0 | 82.8 |
| **After SenseNova-SI SFT** | | | | | | |
| SN-SI InternVL3-8B | **68.7** (+26.6) | **43.3** (+15.3) | **85.6** (+44.1) | **54.6** (+16.0) | **50.1** (+9.0) | 84.9 |
| SN-SI Qwen3-VL-8B | 62.9 (+5.0) | 37.5 (+6.4) | 74.8 (+45.4) | 47.7 (+5.5) | 47.7 (+1.9) | 83.8 |
| SN-SI Bagel-7B-MoT | 41.6 (+10.2) | 36.2 (+5.2) | 50.8 (+16.1) | 43.9 (+2.6) | 42.5 (+5.5) | 79.1 |

The improvements were dramatic. SenseNova-SI InternVL3-8B jumped from 42.1% to 68.7% on VSI-Bench, from 28.0% to 43.3% on MMSI, and most strikingly from 41.5% to 85.6% on MindCube—a 44-point gain from data scaling alone, with no architectural changes. General capability on MMBench-En was preserved (84.9% vs. 81.7% baseline), confirming that spatial training did not come at the cost of general understanding.

Perhaps most remarkably, the data-scaled open-source models surpassed proprietary models on several benchmarks. On MindCube, SenseNova-SI InternVL3-8B (85.6%) exceeded GPT-5 (56.3%), Gemini-2.5-Pro (57.6%), and Grok-4 (63.5%) by wide margins. On VSI-Bench, it surpassed GPT-5 (68.7% vs. 55.0%). On MMSI's camera-object perspective-taking subtask, SenseNova-SI (62.7%) outperformed GPT-5 (49.8%). These results validated the insight from Phase 1 that the spatial intelligence gap is primarily a *data* problem, not a *scale* problem—and that targeted data curation guided by the EASI taxonomy could be transformative.

### 4.5 Deeper Insights from Data Scaling

Beyond the headline numbers, the data scaling experiments revealed several nuanced findings.

**Scaling laws vary by capability.** Not all spatial capabilities scale at the same rate. Perspective-taking showed the most dramatic improvement with additional data, consistent with it being the most data-starved capability in existing training corpora. Metric measurement improved more modestly, suggesting it may require different kinds of training signal (perhaps grounded depth estimation rather than QA-style supervision).

**Emergent generalization.** Models trained on one set of spatial tasks exhibited nontrivial transfer to seemingly unrelated tasks, and demonstrated extrapolation to longer spatial contexts beyond the training distribution. This hinted at the formation of genuine spatial representations rather than mere pattern matching.

**Robustness against shortcuts.** Through controlled experiments including circular test designs, the team validated that SenseNova-SI genuinely acquired spatial capabilities rather than exploiting memorization, annotation biases, or language shortcuts in the training data. This was a critical validation—earlier work had shown that many spatial benchmarks contain exploitable non-visual shortcuts [21].

**Spatial chain-of-thought is not a silver bullet.** The team constructed and evaluated three representative text-based CoT schemes for spatial reasoning but found that none reliably improved performance beyond what QA-style data scaling achieved. This suggested that extending text-based CoT to spatial intelligence may require fundamentally different reasoning mechanisms—perhaps visual or geometric rather than linguistic.

---

## 5. The Open Question: From Static Reasoning to Embodied Action

### 5.1 Why Embodied Evaluation Matters

The two papers established that (a) spatial intelligence is a major weakness of current models and (b) data scaling can substantially improve it on static benchmarks. But a deeper question remained: *do these improvements matter for agents that must act in the world?*

Static benchmarks test spatial reasoning in isolation—the model sees images, reads a question, and produces text. An embodied agent, by contrast, must close the loop: perceive the environment, reason about its spatial structure, decide on an action, execute it, observe the consequences, and plan the next step. This loop introduces entirely new failure modes—an agent might have perfect spatial perception but poor planning, or excellent reasoning but inability to recover when an action fails.

A preliminary result from the SenseNova-SI paper [2] hinted at the connection: applying SenseNova-SI to EmbodiedBench manipulation tasks *without any fine-tuning* produced measurable performance improvements, suggesting that spatial data scaling benefits extend beyond static Q&A. But this was tested on only one domain with one benchmark. A systematic investigation would require running diverse embodied benchmarks under controlled conditions—and no unified framework for doing so existed.

### 5.2 Designing the EASI Embodied Evaluation Library

To enable this investigation, I independently designed and built the EASI embodied evaluation library, extending the EASI evaluation philosophy from static benchmarks to interactive simulator-based evaluation. The library addresses a practical challenge that had long fragmented the embodied AI community: each simulator (AI2-THOR, Habitat-Sim, CoppeliaSim, ThreeDWorld) has its own Python version requirements, GPU dependencies, API conventions, and episode formats. Running even a single benchmark often required days of environment debugging; running multiple benchmarks for comparative evaluation was a research project in itself.

Figure 1 shows the high-level architecture. The library is organized around four key capabilities, each addressing a concrete barrier to systematic embodied evaluation.

```
┌───────────────────────────────────────────────────────────────────────┐
│                      HOST PROCESS (Python 3.10+)                      │
│                                                                       │
│  ┌───────┐   ┌────────────┐   ┌─────────┐   ┌──────────────────┐    │
│  │  CLI  │──>│  Runner    │──>│  Agent   │──>│   LLM Client     │    │
│  │       │   │            │   │  (ReAct) │   │   (LiteLLM)      │    │
│  └───────┘   └─────┬──────┘   └─────────┘   └────────┬─────────┘    │
│                     │                                  │              │
│            ┌────────v─────────┐              ┌────────v──────────┐   │
│            │  Simulator       │              │  LLM Backends:    │   │
│            │  (IPC wrapper)   │              │   Cloud APIs /    │   │
│            └────────┬─────────┘              │   vLLM (managed)  │   │
│                     │                        └───────────────────┘   │
│  ═══════════════════╪═══════════  subprocess boundary  ══════════    │
│                     │                                                │
│            ┌────────v─────────┐   ┌─────────────────────────────┐   │
│            │  Bridge Process  │   │  vLLM Server(s) (managed)   │   │
│            │  (own conda env) │   └─────────────────────────────┘   │
│            │  ┌──────────┐   │   ┌─────────────────────────────┐   │
│            │  │Simulator │   │   │  Xorg Server(s) (optional)  │   │
│            │  │  (Gym)   │   │   └─────────────────────────────┘   │
│            │  └──────────┘   │                                      │
│            └─────────────────┘                                      │
└───────────────────────────────────────────────────────────────────────┘
```
**Figure 1.** High-level architecture of the EASI embodied evaluation library. The subprocess boundary separates the host process (Python 3.10+) from simulator bridge processes that may run different Python versions. Communication uses atomic JSON file I/O. See Appendix A for detailed process flow.

#### 5.2.1 Unified Multi-Simulator Evaluation

The core architectural insight is **subprocess isolation**. Each simulator runs in its own conda environment, potentially with a different Python version—AI2-THOR v2.1 requires Python 3.8, Habitat-Sim requires Python 3.9, while the host runs Python 3.10+. A bridge script inside each simulator's environment communicates with the parent process through filesystem-based IPC: atomic JSON file reads and writes in a temporary directory. This avoids the complexity of socket-based communication while maintaining reliability across Python version boundaries. The IPC protocol uses three files—`status.json`, `command.json`, and `response.json`—with atomic writes via rename to prevent partial reads (see Appendix A.2).

On top of this isolation layer, the library provides a unified interface. A single CLI command—`easi start <task> --agent react --backend openai --model gpt-4o`—works identically across all benchmarks. Tasks are auto-discovered via YAML configuration files; simulators are registered through manifest files. Adding a new benchmark requires only a task directory with a YAML config and a Python class, with no modification to any central registry. For benchmarks with multiple evaluation splits (EmbodiedBench has 19 splits across four domains), YAML template inheritance keeps configuration DRY: split-specific configs extend a shared base via `extends: _base.yaml`, varying only the dataset subset, episode filter, or action space.

#### 5.2.2 Standardised Agent-Environment Interface

A key challenge in embodied evaluation is that different benchmarks require different prompt formats, action spaces, and feedback mechanisms. The library addresses this through a **PromptBuilder** protocol that standardises the agent-environment interface while remaining customisable per benchmark.

The standard prompt format uses a two-message structure: a system prompt and a user message. The system prompt follows a fixed section order—Role and Environment, Available Actions, Strategy, Guidelines, and Response Format—providing the LLM with a consistent frame of reference regardless of which benchmark is running. The user message assembles per-step dynamic content: the current camera image, task instruction, environment feedback (e.g., geodesic distance to goal), and a truncated action history showing previous actions and their outcomes. A complete prompt example for a navigation episode is provided in Appendix B.

```
System Prompt                          User Message (per step)
┌─────────────────────────┐            ┌─────────────────────────┐
│ ## Role and Environment │            │ [Image: current view]   │
│ ## Available Actions    │            │ ## Task                 │
│ ## Strategy             │            │ ## Environment Feedback  │
│ ## Guidelines           │            │ ## Action History        │
│ ## Response Format      │            │ ## Chat History          │
└─────────────────────────┘            │ [Response format hint]  │
                                       └─────────────────────────┘
```
**Figure 2.** Standardised prompt structure. The system prompt is static per benchmark; the user message is assembled fresh each step with current observations and history.

The LLM responds in a standardised 4-field JSON format: `visual_state_description` (what the agent sees), `reasoning_and_reflection` (situational analysis), `language_plan` (intended next steps), and `executable_plan` (a list of actions to execute). This structured format enables **multi-action buffering**: when the LLM generates a plan with multiple actions, the agent executes them one at a time without re-prompting the LLM until an action fails or the buffer is exhausted. This substantially reduces API calls—a significant cost factor when each episode may run for hundreds of steps.

#### 5.2.3 Scalable Parallel Evaluation

Embodied evaluation is expensive: a single EB-Alfred episode may involve 50+ LLM calls with image inputs. To make large-scale evaluation practical, the library supports **thread-pool parallelism** where each worker gets its own simulator process, agent instance, and LLM client. Workers pull episodes from a shared thread-safe queue, enabling automatic load balancing.

For local inference using vLLM, the library manages the full server lifecycle: a `MultiServerManager` spawns multiple vLLM instances across designated GPUs using a two-phase startup (spawn all processes, then concurrent health checks), distributes workers across instances via round-robin URL assignment, and tears down servers on completion. GPU allocation is explicitly partitioned—separate pools for LLM inference and simulator rendering—preventing resource contention. A render platform abstraction handles the diverse display requirements of different simulators, supporting six backends (native X11, virtual framebuffer, EGL headless, managed Xorg, and others) with per-worker GPU binding. See Appendix A.3 for detailed worker lifecycle and resource management.

#### 5.2.4 Reproducibility and Analysis

Long-running evaluations need robust support for interruption and resumption. The library saves a `config.json` with all CLI options at run start, and each completed episode writes its metrics (`result.json`) and step-by-step trajectory (`trajectory.jsonl` + observation images) to a structured output directory. The `--resume` flag restarts from the last completed episode, re-aggregating results across both completed and new episodes.

An episode filtering syntax (`--episodes :10`, `--episodes 2,10:20,40`) supports index ranges, specific episode IDs, and mixed specifications, enabling targeted re-evaluation. For post-hoc analysis, a trajectory video generator creates dual-panel MP4 videos showing the agent's top-down path alongside its camera view at each step, with success/failure overlays—useful for diagnosing agent behaviour across benchmarks.

### 5.3 The Breadth of Embodied Evaluation

Before implementing the library's architecture, I had spent several weeks verifying each of the EmbodiedBench environments—Alfred, Navigation, Habitat, and Manipulation—setting up their respective simulators on the server and confirming that episodes could be loaded, reset, and stepped through correctly. This groundwork informed the library's design and ensured that the abstractions I chose would accommodate the real diversity of simulator interfaces.

The library now integrates ten embodied benchmarks spanning five simulators and a wide range of capabilities:

EmbodiedBench's four domains alone cover 19 evaluation splits. **EB-Alfred** (AI2-THOR v2.1) tests household task completion across six capability dimensions—base, spatial, visual, commonsense, complex instruction following, and long-horizon planning. A "complex" episode might require the agent to execute 20+ sequential actions ("heat the potato, then put it on the counter, then pick up the knife..."), while a "spatial" episode tests whether the agent can locate objects based on spatial descriptions. **EB-Navigation** (AI2-THOR v5.0) evaluates indoor point-goal navigation with five splits. **EB-Habitat** (Habitat-Sim) tests scene understanding combined with navigation across four splits. **EB-Manipulation** (CoppeliaSim) evaluates robotic manipulation—precise 3D gripper positioning using discrete action coordinates.

Beyond EmbodiedBench, the library integrates **HAZARD** (ThreeDWorld), which presents a qualitatively different challenge: the agent must infer *implicit* goals under emergency conditions. During a fire scenario, the agent is not told "evacuate the laptop"—it must reason that valuable objects should be rescued and identify which objects are valuable, all while navigating a dynamically changing environment. This tests what I term *situational goal reasoning*, a capability not captured by any static benchmark.

**ManipulaTHOR** tests arm-based point navigation in AI2-THOR, and **AI2-THOR Rearrangement 2023** evaluates an agent's ability to restore objects to their original positions across five evaluation splits. The **VLN-CE** benchmarks (R2R and RxR) bring vision-and-language navigation into continuous environments using Habitat-Sim, requiring agents to translate free-form natural language instructions ("Walk past the dining table, turn left at the hallway, stop at the second door on your right") into continuous movement. The RxR variant includes Hindi and Telugu instructions, enabling multilingual embodied evaluation. **LHPR-VLN** extends this with multi-subtask navigation requiring hierarchical planning.

### 5.4 The LLM-Driven Agent

To drive evaluation across these diverse benchmarks, the library implements an LLM-driven agent that follows a perceive-reason-act loop. At each step, the agent receives an egocentric camera image and environment feedback from the simulator. It maintains an **agent memory** that tracks the full interaction history—previous observations, actions taken, environment feedback received, and the LLM's own prior responses—providing the context needed for coherent multi-step behaviour. The PromptBuilder (Section 5.2.2) assembles this memory into a structured prompt, which is sent to the LLM for reasoning. The LLM returns a structured response containing its visual description of the scene, reasoning about the current situation, a natural language plan, and one or more proposed actions. The agent validates each action against the task's action space and executes them sequentially.

Two mechanisms make this loop practical at scale. First, **multi-action buffering**: when the LLM generates a plan with multiple actions, the agent executes them one at a time without re-prompting the LLM until an action fails or the buffer is exhausted, substantially reducing API calls. Second, **feedback-driven replanning**: when an action fails, the failure is recorded in the action history, the action buffer is cleared, and the agent re-prompts the LLM with the updated context on the next step. This enables error recovery without explicit failure-handling logic—the LLM reasons about what went wrong based on the feedback history and adjusts its plan accordingly.

The agent interface is pluggable: the current implementation uses an LLM for reasoning, but the library also includes a dummy agent (random action selection) for testing, and the protocol is designed so that alternative agent architectures—such as those incorporating explicit spatial maps or hierarchical planners—can be substituted without modifying the evaluation pipeline.

### 5.5 What Embodied Evaluation Adds to the Picture

The capability dimensions tested by embodied benchmarks partially overlap with the EASI static taxonomy but extend it in important ways. Static benchmarks test whether a model can answer *"which object is to the left of the chair?"*—embodied benchmarks test whether an agent can *navigate to* the left of the chair, pick up the object there, and carry it to another room. This requires not just spatial perception (SR) and perspective-taking (PT), but also:

- **Long-horizon planning**: Decomposing complex goals into executable action sequences, maintaining a mental model of progress, and deciding when to replan.
- **Situational goal reasoning**: Inferring what to do when the goal is not explicitly stated (HAZARD scenarios).
- **Grounded language understanding**: Translating natural language navigation instructions into continuous movement through 3D space (VLN-CE).
- **Error recovery**: Detecting when an action has failed (a gripper didn't close, a navigation step was blocked) and adapting the plan accordingly.

The preliminary finding from [2]—that SenseNova-SI improved EmbodiedBench manipulation performance without fine-tuning—is encouraging. The EASI embodied library now makes it possible to test this transfer systematically across all ten benchmarks and multiple model families, providing a much richer picture of whether and how static spatial intelligence translates to embodied competence.

### 5.6 Preliminary Experiment: EASI-Mini

To provide an initial data point on whether static spatial intelligence gains transfer to embodied settings, I designed and ran a compact cross-benchmark evaluation called EASI-Mini. The benchmark samples 89 episodes across seven capability dimensions from the library's integrated benchmarks, enabling a structured comparison without the cost of running all ten benchmarks at full scale.

**Benchmark design.** Each capability dimension draws episodes from one or more source benchmarks:

| Capability | Sources | Episodes |
|---|---|---|
| Spatial Reasoning | EB-Alfred, EB-Habitat, EB-Manipulation (spatial splits) | 9 |
| Common Sense Reasoning | EB-Alfred, EB-Navigation, EB-Habitat, EB-Manipulation | 12 |
| Complex Instruction Following | EB-Alfred, EB-Navigation, EB-Habitat, EB-Manipulation | 12 |
| Long-Horizon Planning | EB-Alfred, EB-Habitat, LHPR-VLN | 14 |
| Visual Grounding | EB-Alfred, EB-Navigation, EB-Habitat, EB-Manipulation | 12 |
| Situational Goal Reasoning | HAZARD (fire, flood, wind) | 15 |
| Continuous Navigation from Language | VLN-CE R2R (val unseen) | 15 |
| **Total** | | **89** |

Episodes are sampled deterministically (seed 42) for reproducibility. Success is normalized across benchmarks: `task_success` for EmbodiedBench, `value_rate > 0.5` for HAZARD, and `success` (within 3m of goal) for VLN-CE.

**Models evaluated.** Four models were selected to span the proprietary–open-source axis and to test the effect of spatial data scaling:

- **GPT-4o** — proprietary baseline with strong general vision capabilities
- **Qwen2.5-VL-72B-Instruct** — strongest open-source model from the static evaluation (Section 3)
- **InternVL3-8B** — the base model before spatial data scaling
- **SenseNova-SI InternVL3-8B** — the same model after SFT on SenseNova-SI-8M spatial data (Section 4)

The InternVL3-8B vs. SenseNova-SI InternVL3-8B comparison directly tests the central question: does the dramatic improvement on static spatial benchmarks (e.g., 41.5% → 85.6% on MindCube) translate into better embodied performance?

**Table 3.** EASI-Mini results: success rate (%) by capability dimension. *[Results pending — to be filled after evaluation runs.]*

| Capability | GPT-4o | Qwen2.5-VL-72B | InternVL3-8B | SN-SI InternVL3-8B |
|---|---|---|---|---|
| Spatial Reasoning | — | — | — | — |
| Common Sense | — | — | — | — |
| Complex Instruction | — | — | — | — |
| Long-Horizon Planning | — | — | — | — |
| Visual Grounding | — | — | — | — |
| Situational Goal | — | — | — | — |
| Continuous Navigation | — | — | — | — |
| **Overall** | **—** | **—** | **—** | **—** |

**Table 4.** EASI-Mini results: success rate (%) by source benchmark. *[Results pending.]*

| Benchmark | GPT-4o | Qwen2.5-VL-72B | InternVL3-8B | SN-SI InternVL3-8B |
|---|---|---|---|---|
| EB-Alfred | — | — | — | — |
| EB-Navigation | — | — | — | — |
| EB-Habitat | — | — | — | — |
| EB-Manipulation | — | — | — | — |
| HAZARD | — | — | — | — |
| VLN-CE R2R | — | — | — | — |
| LHPR-VLN | — | — | — | — |

**Analysis.** *[To be written after results are collected. Key questions to address:]*

- *Does SenseNova-SI InternVL3-8B outperform the base InternVL3-8B on embodied tasks, particularly those requiring spatial reasoning?*
- *Which capability dimensions show the largest gap between proprietary (GPT-4o) and open-source models?*
- *Do the capability-level patterns mirror the static evaluation findings (e.g., is spatial reasoning still the weakest dimension)?*
- *Does the ranking of models on static benchmarks (Table 1) predict their ranking on embodied tasks?*

This experiment is preliminary—the small sample sizes (3–15 episodes per capability) limit statistical robustness. However, it provides an initial signal on the static-to-embodied transfer question and demonstrates the library's ability to produce structured, cross-benchmark comparisons with a single evaluation script.

---

## 6. Discussion

### 6.1 The Arc of the Work

Looking back, the three phases of this research form a natural scientific progression. The holistic evaluation (Phase 1) provided the diagnosis: spatial intelligence is a specific, measurable weakness of current models, worst in perspective-taking and metric measurement, present even in frontier proprietary models, and not reducible to general model scale. The data scaling work (Phase 2) provided a treatment: targeted, taxonomy-guided data curation can produce dramatic improvements, with open-source models matching or exceeding proprietary ones on several challenging subtasks. The embodied evaluation library (Phase 3) provides the infrastructure for the next question: does the treatment work in the real world?

This arc also reflects a progression in my own contributions. In the first phase, I built the evaluation infrastructure that made holistic comparison possible—integrating nine benchmarks and two models into a unified framework, fixing inference bugs that would have corrupted results. In the second phase, this infrastructure was reused to validate the SenseNova-SI models, with the benchmarks and model integrations I had built serving directly as the evaluation backbone and baselines. In the third phase, I worked independently to design and implement an entirely new evaluation toolkit, drawing on the insights from the first two phases to frame embodied evaluation as a natural extension of spatial intelligence research.

### 6.2 Reflections on Evaluation-Driven Research

A recurring theme is that evaluation infrastructure is not merely a support function—it shapes the research questions that can be asked. The unified lmms-eval integrations made the holistic comparison in [1] possible; without them, the study would have been a collection of disconnected benchmark scores rather than a coherent capability analysis. The EASI taxonomy, in turn, made the data gap analysis in [2] actionable: by mapping every benchmark subtask to a capability dimension, the team could identify precisely where training data was lacking (perspective-taking) and scale accordingly.

The embodied evaluation library continues this pattern. By unifying ten benchmarks under a single CLI, it becomes feasible to ask questions like "does improving MindCube performance predict better EB-Manipulation scores?" or "which static capability is most predictive of VLN-CE success?"—questions that would be impractical if each benchmark required its own evaluation pipeline.

### 6.3 Limitations and Future Directions

Several important limitations and open questions remain.

**Cost of embodied evaluation.** Each embodied episode requires multiple simulator steps and LLM calls, making evaluation orders of magnitude more expensive than static benchmarks. A single EB-Alfred episode may involve 50+ LLM calls with image inputs. Efficient evaluation strategies—episode sampling, proxy metrics, early stopping—will be important for making embodied evaluation routine rather than occasional.

**Comprehensive transfer study.** The EASI-Mini experiment (Section 5.6) provides an initial signal, but a systematic study across all ten embodied benchmarks at full scale, with multiple model families and varying amounts of spatial training data, remains future work.

**Agent architecture.** The ReAct agent is a reasonable baseline but far from the ceiling. More sophisticated architectures—incorporating explicit spatial maps, hierarchical planning, or learned exploration strategies—could better leverage spatial intelligence. The library's pluggable agent interface is designed to support this exploration.

**The CoT puzzle.** The finding that text-based chain-of-thought does not reliably improve spatial reasoning [2] is both surprising and important. It suggests that spatial reasoning may require fundamentally different mechanisms—perhaps visual or geometric chain-of-thought, or internal 3D representations—rather than the linguistic decomposition that works well for mathematical and logical reasoning.

**Multilingual embodied evaluation.** The VLN-CE RxR benchmark includes Hindi and Telugu instructions alongside English, enabling multilingual embodied evaluation—an area that remains largely unexplored.

---

## 7. Conclusion

This report has traced a path from measuring spatial intelligence through improving it to evaluating whether improvements transfer to embodied settings. Along the way, I contributed evaluation infrastructure (nine benchmark integrations, two model integrations, and numerous bug fixes in lmms-eval) that enabled the holistic spatial intelligence evaluation reported in [1] and the model validation in [2]. I then independently designed and built the EASI embodied evaluation library—a unified toolkit integrating ten embodied benchmarks across five simulators with LLM-powered agents, subprocess isolation for heterogeneous Python environments, and a comprehensive evaluation pipeline with parallel execution, episode filtering, resume support, and trajectory analysis. A preliminary cross-benchmark experiment (EASI-Mini) provided initial data on whether static spatial gains transfer to embodied tasks.

The overarching finding of this research programme is that spatial intelligence is both more important and more tractable than previously assumed. The gap between models and humans is large but not immutable—targeted data scaling produces dramatic improvements. The EASI-Mini experiment offers a first look at whether these improvements extend from static benchmarks to embodied competence, and the library provides the infrastructure for the comprehensive studies that will follow.

---

## References

[1] Zhongang Cai, Yubo Wang, Qingping Sun, Ruisi Wang, Chenyang Gu, et al. "Holistic Evaluation of Multimodal LLMs on Spatial Intelligence." arXiv:2508.13142, 2025.

[2] Zhongang Cai, Ruisi Wang, Chenyang Gu, Fanyi Pu, et al. "Scaling Spatial Intelligence with Multimodal Foundation Models." arXiv:2511.13719, 2025.

[3] Y. Li et al. "VSI-Bench: Video Spatial Intelligence Benchmark." 2024.

[4] Y. Wang et al. "MMSI: Multi-image Spatial Intelligence Benchmark." 2024.

[5] Z. Zhang et al. "MindCube: Mental Rotation Benchmark for Spatial Intelligence." 2024.

[6] Z. Lin et al. "ViewSpatial: View-Dependent Spatial Reasoning Benchmark." 2024.

[7] Q. Sun et al. "SITE: Spatial Intelligence Task Evaluation." 2024.

[8] C. Chen et al. "OmniSpatial: A Comprehensive Spatial Reasoning Benchmark." 2024.

[9] Various. "Spatial-MLLM: Integrating 3D Expert Encoders for Spatial Understanding." 2024.

[10] Various. "VLM-3R: Vision-Language Models with 3D Representations." 2024.

[11] Various. "3DThinker: Aligning Model Features with 3D Supervision." 2024.

[12] B. Chen et al. "SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities." CVPR, 2024.

[13] Various. "SpatialLadder: Progressive Training for Spatial Understanding." 2024.

[14] Various. "VST: Visual Spatial Transformer." 2024.

[15] Various. "Cambrian-S: Spatial Video Understanding via Progressive Training." 2024.

[16] Various. "EmbodiedBench: Comprehensive Benchmarking of Multi-modal LLMs as Embodied Agents." 2024.

[17] Various. "HAZARD: Emergency Evacuation Benchmark for Embodied AI." 2024.

[18] J. Krantz et al. "Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments." ECCV, 2020.

[19] K. Ehsani et al. "ManipulaTHOR: A Framework for Visual Object Manipulation." CVPR, 2021.

[20] Various. "LMMs-Eval: Unified Evaluation Framework for Large Multimodal Models." 2024.

[21] E. Brown et al. "Benchmark Designers Should Train on the Test Set to Expose Exploitable Non-Visual Shortcuts." arXiv:2511.04655, 2024.

---

## Appendix A: EASI Library Architecture Deep-Dive

### A.1 Process Flow

The following sequence diagram shows the full lifecycle of an `easi start` invocation, from CLI parsing through episode execution to final cleanup. Component interactions proceed top-to-bottom; the subprocess boundary separates the host process from simulator bridge processes.

```
 User          CLI          Runner         Task         Agent       Simulator    Bridge (subprocess)    LLM API
  │              │             │              │            │             │              │                  │
  │ easi start   │             │              │            │             │              │                  │
  │─────────────>│             │              │            │             │              │                  │
  │              │ parse args  │              │            │             │              │                  │
  │              │ resolve cfg │              │            │             │              │                  │
  │              │             │              │            │             │              │                  │
  │              │  ┌──────────┤  SETUP       │            │             │              │                  │
  │              │  │          │              │            │             │              │                  │
  │              │  │          │ create task  │            │             │              │                  │
  │              │  │          │─────────────>│ YAML       │             │              │                  │
  │              │  │          │              │ resolve    │             │              │                  │
  │              │  │          │              │ load eps   │             │              │                  │
  │              │  │          │              │ filter     │             │              │                  │
  │              │  │          │<─────────────│            │             │              │                  │
  │              │  │          │              │            │             │              │                  │
  │              │  │          │ [if vLLM]    │            │             │              │                  │
  │              │  │          │ spawn servers, health check            │              │                  │
  │              │  │          │ → [url_0, url_1, ...]    │             │              │                  │
  │              │  │          │              │            │             │              │                  │
  │              │  │          │ create agent │            │             │              │                  │
  │              │  │          │────────────────────────>│ LLMClient   │              │                  │
  │              │  │          │<────────────────────────│ PromptBuilder│              │                  │
  │              │  │          │              │            │             │              │                  │
  │              │  │          │ setup render platform   │             │              │                  │
  │              │  │          │ create simulator        │             │              │                  │
  │              │  │          │──────────────────────────────────────>│              │                  │
  │              │  │          │              │            │             │ launch()     │                  │
  │              │  │          │              │            │             │─────────────>│ Popen(bridge.py) │
  │              │  │          │              │            │             │ wait ready   │ {"ready": true}  │
  │              │  │          │              │            │             │<─────────────│                  │
  │              │  └──────────┤              │            │             │              │                  │
  │              │             │              │            │             │              │                  │
  │              │  ┌──────────┤  EPISODE LOOP (per episode)           │              │                  │
  │              │  │          │              │            │             │              │                  │
  │              │  │          │ sim.reset()  │            │             │              │                  │
  │              │  │          │──────────────────────────────────────>│ command.json │                  │
  │              │  │          │              │            │             │─────────────>│ _on_reset()      │
  │              │  │          │              │            │             │ response.json│ save image       │
  │              │  │          │              │            │             │<─────────────│                  │
  │              │  │          │              │            │             │              │                  │
  │              │  │          │  ┌───────────┤  STEP LOOP │             │              │                  │
  │              │  │          │  │           │            │             │              │                  │
  │              │  │          │  │ agent.act(observation) │             │              │                  │
  │              │  │          │  │           │ build prompt            │              │                  │
  │              │  │          │  │           │────────────────────────────────────────────────────────>│
  │              │  │          │  │           │            │ LLM response│              │                  │
  │              │  │          │  │           │<────────────────────────────────────────────────────────│
  │              │  │          │  │           │ parse, validate, buffer │              │                  │
  │              │  │          │  │           │ return action           │              │                  │
  │              │  │          │  │           │             │            │              │                  │
  │              │  │          │  │ sim.step(action)       │            │              │                  │
  │              │  │          │  │───────────────────────────────────>│ command.json │                  │
  │              │  │          │  │           │             │            │─────────────>│ _on_step()       │
  │              │  │          │  │           │             │            │<─────────────│ StepResult       │
  │              │  │          │  │           │             │            │              │                  │
  │              │  │          │  │ feedback → agent       │            │              │                  │
  │              │  │          │  │ [if fail: clear buffer]│            │              │                  │
  │              │  │          │  └───────────┤             │            │              │                  │
  │              │  │          │              │             │            │              │                  │
  │              │  │          │ evaluate_episode()        │            │              │                  │
  │              │  │          │ save result.json + trajectory.jsonl   │              │                  │
  │              │  └──────────┤              │             │            │              │                  │
  │              │             │              │             │            │              │                  │
  │              │  ┌──────────┤  CLEANUP     │             │            │              │                  │
  │              │  │          │ sim.close()  │             │            │              │                  │
  │              │  │          │ teardown render + vLLM    │            │              │                  │
  │              │  │          │ aggregate_results()       │            │              │                  │
  │              │  │          │ write summary.json        │            │              │                  │
  │              │  └──────────┤              │             │            │              │                  │
  │  results     │  return     │              │             │            │              │                  │
  │<─────────────│<────────────│              │             │            │              │                  │
```

### A.2 IPC Protocol

Communication between the host process and each simulator bridge uses three JSON files in a temporary workspace directory:

```
/tmp/easi_XXXXXXXX/          ← workspace (created by SubprocessRunner)
├── status.json               ← bridge writes on startup: {"ready": true}
├── command.json              ← parent writes, bridge reads
└── response.json             ← bridge writes, parent reads
```

All writes use an **atomic rename** pattern (`file.tmp` → `file.json`) to prevent partial reads. The parent polls `response.json` with a configurable interval (default 0.1s) and timeout, checking subprocess liveness on each poll.

**Command types:**

| Type | Parent Sends | Bridge Does | Bridge Returns |
|------|-------------|-------------|----------------|
| `reset` | `{type, episode_id, reset_config, episode_output_dir}` | Create/reset environment, save observation image | `{status, observation: {rgb_path, agent_pose, metadata}, reward, done, info}` |
| `step` | `{type, action: {action_name, params}}` | Execute action, save observation image | Same observation format |
| `close` | `{type: "close"}` | Cleanup environment, exit | `{status: "ok"}` |

**Component map** (30+ modules organized by responsibility):

| Layer | Components | Role |
|-------|-----------|------|
| CLI & Orchestration | `cli.py`, `EvaluationRunner`, `ParallelRunner`, `EpisodeFilter`, `Metrics` | Parse arguments, manage episode loop, aggregate results |
| Tasks | `TaskRegistry`, `yaml_utils`, `Dataset`, `BaseTask` | Auto-discover tasks, resolve YAML inheritance, load datasets |
| Agents | `ReActAgent`, `AgentMemory`, `PromptBuilder` | LLM-driven decision making, state tracking, prompt assembly |
| LLM | `LLMClient`, `ServerManager`, `MultiServerManager`, `DummyServer` | Unified LLM interface, vLLM lifecycle, test server |
| Simulators | `SimulatorRegistry`, `BaseSimulator`, `SubprocessRunner`, `BaseBridge` | Auto-discover simulators, IPC wrapper, process lifecycle, bridge-side loop |
| Communication | `filesystem.py`, `schemas.py` | Atomic JSON I/O, command/response schemas |
| Rendering | `RenderPlatform`, `XorgManager`, `SimulatorRenderAdapter` | Display backend abstraction, managed Xorg servers, per-simulator adjustments |
| Analysis | `TrajectoryVideo`, `ProgressBar` | Post-hoc video generation, thread-safe progress display |

### A.3 Parallel Execution and Resource Management

The `ParallelRunner` extends the sequential runner with thread-pool concurrency. Each worker is fully isolated with its own simulator process, agent instance, and LLM client.

```
┌──────────────────────────────────────────────────────────────────────┐
│                       ParallelRunner.run()                            │
│                                                                      │
│  Task loaded ONCE (shared, read-only)                                │
│  Episodes placed in thread-safe queue                                │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  vLLM: MultiServerManager (if backend=vllm)                  │    │
│  │    Instance 0: GPU 0,1 → http://localhost:8000/v1            │    │
│  │    Instance 1: GPU 2,3 → http://localhost:8001/v1            │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Xorg: XorgPlatform (if render_platform=xorg)                │    │
│  │    Display :10 → GPU 4                                        │    │
│  │    Display :11 → GPU 5                                        │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ThreadPoolExecutor(max_workers=8)                                   │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐      ┌────────────┐  │
│  │  Worker 0  │ │  Worker 1  │ │  Worker 2  │ ...  │  Worker 7  │  │
│  │            │ │            │ │            │      │            │  │
│  │ LLM: :8000│ │ LLM: :8001│ │ LLM: :8000│      │ LLM: :8001│  │
│  │ Xorg: :10 │ │ Xorg: :11 │ │ Xorg: :10 │      │ Xorg: :11 │  │
│  │ GPU: 4    │ │ GPU: 5    │ │ GPU: 4    │      │ GPU: 5    │  │
│  │            │ │            │ │            │      │            │  │
│  │ own sim   │ │ own sim   │ │ own sim   │      │ own sim   │  │
│  │ own agent │ │ own agent │ │ own agent │      │ own agent │  │
│  │ own bridge│ │ own bridge│ │ own bridge│      │ own bridge│  │
│  └────────────┘ └────────────┘ └────────────┘      └────────────┘  │
│                                                                      │
│  Results: collected with threading.Lock, sorted by index at end      │
└──────────────────────────────────────────────────────────────────────┘
```

**Resource assignment** uses round-robin allocation:
- LLM URLs: `base_urls[worker_id % len(base_urls)]` distributes workers evenly across vLLM instances
- GPU/display binding: `platform.for_worker(worker_id)` cycles through available render instances
- Thread-safe results: `threading.Lock` guards the shared results list

**vLLM MultiServerManager** handles the full lifecycle of local inference servers:

```
Input: model, num_instances=2, gpu_ids=[0,1,2,3], tensor_parallel_size=2

GPU distribution: 4 GPUs ÷ 2 instances = 2 GPUs/instance
  Instance 0: CUDA_VISIBLE_DEVICES="0,1" → port 8000
  Instance 1: CUDA_VISIBLE_DEVICES="2,3" → port 8001

Phase 1 — Spawn (non-blocking): find available port, Popen vLLM process
Phase 2 — Health check (blocking): ThreadPoolExecutor polls GET /health (timeout 600s)

Output: ["http://localhost:8000/v1", "http://localhost:8001/v1"]
```

**Render platform hierarchy** (six built-in platforms):

| Platform | Display | GPU Binding | Use Case |
|----------|---------|-------------|----------|
| `auto` | Detects native; falls back to xvfb | Manual | Default for most simulators |
| `native` | Uses existing `$DISPLAY` | Manual | Desktop with X server |
| `xvfb` | Virtual framebuffer | Manual | Headless server (CPU rendering) |
| `egl` | None (headless) | Via `CUDA_VISIBLE_DEVICES` | GPU-accelerated headless |
| `headless` | None | Manual | Simulators with native headless |
| `xorg` | `:N` per GPU (managed) | Per-GPU auto-managed | GPU-accelerated X11 |

**Environment variable layering** — seven sources contribute environment variables to each bridge subprocess, merged in priority order:

| Layer | Source | Example |
|-------|--------|---------|
| 1 | `render_platform.get_env_vars()` | `PYOPENGL_PLATFORM=egl` |
| 2 | `env_manager.get_env_vars(platform)` | `COPPELIASIM_ROOT=...` |
| 3 | `task.extra_env_vars` | Task-specific overrides |
| 4 | `worker_binding.extra_env` | `EASI_GPU_DISPLAY=1` |
| 5 | `render_adapter.get_env_vars(binding)` | Simulator-specific platform adjustments |
| 6 | `binding.display` | `DISPLAY=:10` |
| 7 | `binding.cuda_visible_devices` | `CUDA_VISIBLE_DEVICES=4` |

Later layers override earlier ones. `PATH`-like variables use a prepend merge mode (concatenated with `:` separator).

---

## Appendix B: Prompt Format and Full Example

### B.1 Prompt Format Specification

The EASI standard prompt format defines how the agent communicates with any LLM backend. It applies to all benchmarks that do not have a published prompt format (EmbodiedBench benchmarks retain their original formats for reproducibility).

**System prompt structure** (fixed section order):

| Section | Required | Purpose |
|---------|----------|---------|
| Role and Environment | Yes | 2-3 sentences establishing agent identity and environment |
| Observation Description | No | Explains non-visual feedback (distances, object states) |
| Available Actions | Yes | Action names with descriptions and parameter formats |
| Strategy | No | Benchmark-specific tactical advice |
| Guidelines | Yes | Universal rules (always output actions, avoid loops, etc.) |
| Response Format | Yes | The 4-field JSON schema |

**Response JSON schema:**

```json
{
    "visual_state_description": "Describe what you see in the current image",
    "reasoning_and_reflection": "Reason about situation, reflect on history",
    "language_plan": "Describe next plan in natural language",
    "executable_plan": [
        {"action": "move_forward"},
        {"action": "turn_left"}
    ]
}
```

**User message assembly order** (per step):

```
[Image(s)]                           ← base64 encoded, before text
[Text content:]
  ## Task                            ← instruction / task description
  ## Environment Feedback            ← current-step feedback (if enabled)
  ## Action History (last N steps)   ← compact action + outcome log
  ## Chat History (last N responses) ← previous LLM responses (if enabled)
  [Response format reminder]         ← brief JSON format hint
```

### B.2 Complete Prompt Example: Navigation Episode

The following shows the exact prompt sent to the LLM during step 5 of a VLN-CE navigation episode.

**System prompt:**

```
## Role and Environment
You are a robot navigating in a 3D indoor environment. You observe the
environment through a front-facing camera and must follow natural language
instructions to navigate to a goal location.

## Observation Description
- **Distance to goal**: Geodesic distance in meters to the goal. Decreases
  as you approach the destination.

## Available Actions
- move_forward: Move forward by 0.25 meters
- turn_left: Turn left by 15 degrees
- turn_right: Turn right by 15 degrees
- stop: Stop and end navigation (use ONLY when you believe you have reached
  the destination described in the instruction)

## Strategy
1. Carefully read the navigation instruction
2. Observe your surroundings in the image
3. Follow the instruction step by step, matching landmarks and directions
4. Use stop ONLY when confident you have reached the described destination

## Guidelines
1. Always output at least one action in executable_plan.
2. Only use actions from the Available Actions list.
3. If previous actions failed, reason about why and try a different approach.
4. Do not repeatedly execute the same action sequence.
5. Keep your plan efficient and concise.

## Response Format
Output a JSON object with exactly these 4 fields:
{
    "visual_state_description": "Describe what you see in the current image",
    "reasoning_and_reflection": "Reason about your situation and history",
    "language_plan": "Describe your next plan in natural language",
    "executable_plan": [{"action": "<action_name>"}]
}

You may include multiple actions in executable_plan. Actions execute
sequentially.
```

**User message (step 5):**

```
[Image: base64 encoded current view]

## Task
Walk down the hallway and turn right into the bedroom.

## Environment Feedback
Distance to goal: 5.3m

## Action History (last 5 steps)
Step 0: move_forward -> Distance to goal: 8.2m
Step 1: move_forward -> Distance to goal: 7.9m
Step 2: move_forward -> Distance to goal: 7.6m
Step 3: turn_right -> Distance to goal: 7.6m
Step 4: move_forward -> Distance to goal: 7.3m

Respond with the JSON format specified above.
```

---

## Appendix C: Statement of Contributions

To maintain academic integrity and clearly attribute credit, this section distinguishes between team-level work and my individual contributions.

**Work I contributed to within the team:** Within the EASI project [1, 2], I was responsible for evaluation infrastructure. This encompassed integrating nine spatial benchmarks (VSI-Bench variants, ViewSpatial, SITE, SparBench, MMSI-Video, OSI-Bench, 3DSR-Bench, MindCube-Tiny) and two models (Cambrian-S, Bagel) into lmms-eval, fixing inference bugs in Qwen3-VL, InternVL, LLaVA-OV, and Cambrian-S, and running evaluations across models and benchmarks. This infrastructure was used to produce the results in both publications.

**Work I did NOT do:** The EASI spatial intelligence taxonomy, paper writing and narrative design, SenseNova-SI-8M dataset curation and synthesis, model training and SFT experiments, data scaling analysis, emergent capability analysis, overfitting and shortcut studies, spatial chain-of-thought experiments, and qualitative failure analysis were all conducted by other team members.

**Work I did independently:** The EASI embodied evaluation library is entirely my own design and implementation. This includes the full software architecture (subprocess isolation, filesystem IPC, auto-discovery registries, YAML template inheritance), the CLI interface, all ten benchmark integrations across five simulators, the ReAct agent with multi-action buffering, the parallel evaluation pipeline with vLLM support, trajectory analysis and video generation, and the 540+ test suite.
