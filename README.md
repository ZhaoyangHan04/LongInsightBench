# LongInsightBench
This repository accompanies our paper LongInsightBench: A Comprehensive Benchmark for Evaluating Omni-Modal Models on Human-Centric Long-Video Understanding.
It includes the benchmark data, source code for benchmark construction and experiments, as well as experimental results.


🗂️ Repository Structure

- ```benchmark_qa/```: The main benchmark data.
  - ```qa/```: Full benchmark, organized into 6 JSON files (one per task).
  - ```qa_subset/```: A 10% random subset of each task (used for evaluation in the paper’s experiments), also organized into 6 JSON files.

- ```data/```: Intermediate data generated during the benchmark construction process.
  - ```caption_result/```: Visual & audio captions generated during preprocessing.
  - ```event_lists/```: Corresponding .json files that provide semantic segmentation summaries (an overall video summary, and timestamps & captions of ecah segment) of each video.
  - ```answer_with_alm/```: Answers produced with Qwen2-audio (ALM), which used for filtering the benchmark QAs.
  - ```answer_with_vlm/```: Answers produced with Qwen2.5-VL (VLM), which used for filtering the benchmark QAs.

- ```experiment/```: All experimental results reported in the paper.
  - ```experiment_main/```: Results of mainstream open- and closed-source OLMs evaluated on the full benchmark.
  - ```multi_modality_input/```: Comparative experiments on the subset benchmark to analyze fusion deficits in current OLMs.
  - ```ablation_study/```: Ablation experiments on open-source OLMs, testing different frame counts to examine sensitivity to input length.

- ```src/```: All source code for benchmark construction and experiments.
  - ```video_filter/```: Filters the initial video dataset based on duration and content richness.
  - ```chunking/```: Performs paragraph-level semantic segmentation of videos.
  - ```caption/```: Generates visual and audio captions for each segment based on the segmentation results.
  - ```qa construction/```: Designs QA tasks for six different types.
  - ```qa_check_and filter/```: Implements a three-step rigorous QA pipeline for quality checking and filtering of generated QA pairs.
  - ```subse/```: Samples a subset of the benchmark for experiments.
  - ```test/```: Runs experiments using different models and different settings on the benchmark.
  - ```evaluation/```: Computes accuracy for model answers.


⚙️ Usage

Before running the code, please **modify all input and output paths** in each script according to your local setup.
Use the examples in the source files as references for expected directory structures and file formats.

If you are running model-based experiments, make sure to:
- Set the **model path or API key** in the corresponding scripts.
- Follow the official installation and configuration instructions **from each model’s repository** to prepare the environment.

For experiments in the ```test/``` directory, run either  ```inference/main.py``` (for Ola-7B) or ```main.py``` (for others) after environment setup.


🧠 AI Assistance Disclosure

In accordance with ARR's official guidelines, we utilized generative AI tools like ChatGPT to assist with minor language refinement and the creation of low-novelty text. 
We ensured that all generated content adhered strictly to ethical standards, maintaining transparency, accuracy, and alignment with research integrity throughout the process.
