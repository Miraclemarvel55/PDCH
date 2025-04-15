# A Multimodal Depression Consultation Dataset of Speech and Text with HAMD-17 Assessments

The global surge in depression rates, notably severe in China with over 95 million affected, underscores a dire public health
issue. This is exacerbated by a critical shortfall in mental health professionals and the inadequate rates of depression diagnosis
and treatment in China, highlighting an urgent call for innovative approaches. The advancement of Artificial Intelligence (AI),
particularly Large Language Models (LLMs), offers a promising solution by improving mental health diagnostics and extending
care to underserved areas. However, there is a lack of real data for reliable training and accurate evaluation of AI models. To
this end, this paper presents a high-quality multimodal depression consultation dataset, namely Parallel Data of Depression
Consultation and Hamilton Depression Rating Scale (PDCH). The dataset is constructed based on high-quality consultations
from Beijing Anding Hospital. It provides audio recording and transcription text (from consultations of 100 patients) in real-world
scenarios (face-to-face consultation), as well as corresponding HAMD-17 scales filled out by professionals. The dataset
contains 100 consultations and the audio length exceeds 2937 minutes. Each of them is about 30-min long with more than 150
dialogue turns. Developed with a commitment to diversity, ethics and privacy, the PDCH dataset aims to fill the gap in mental
health services and enable the creation of more sophisticated and accurate AI models for personalized care. This effort not
only addresses the immediate need for improved early depression diagnostics but also facilitates a more inclusive approach to
mental health care globally.

# PDCH Dataset Evaluation for LLMs and Multimodal Models

This repository contains code for evaluating large language models (LLMs) and multimodal models on depression interview dialogues. The evaluation supports both text-only models and audio-text multimodal models.

## Dataset preprocess and Binning
- Loads and validates 99 depression assessment instances from structured Excel (HAMD-17 scores) and JSON metadata
- Cross-Validation Architecture:
    - Implements stratified data splitting by:
    - Balance metric (imbalance degree)
- Saves three core JSON files:
    - dialogue_messages.json: Complete annotated conversations
    - dialogue_names.json: Basic dialogue identifiers
    - dialogue_names_with_audio_emotions.json: Audio-enhanced subset
- Organizes 5-fold splits in bin_name dir:
```bash
python generate_sft_conversation.py
```

## Features

- Supports both text-only and multimodal (audio+text) models
- Option to include audio emotion annotations for text-only models
- Multiple audio input formats supported
- Flexible configuration for different model types

## Supported Models

### Text-only Models:

- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

### Multimodal Models (Audio+Text):

- `Qwen/Qwen2-Audio-7B-Instruct`
- `gpt-4o-mini-audio-preview`
- `Qwen/Qwen2.5-Omni-7B`

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python eval_generate.py \
    --model_name_or_path <model_name> \
    --modals <modalities> \
    --data_source_path <path_to_data>
```

### Parameters

| Parameter                      | Description                                       | Default                                     | Options                              |
| ------------------------------ | ------------------------------------------------- | ------------------------------------------- | ------------------------------------ |
| `--CUDA_VISIBLE_DEVICES`       | GPU devices to use                                | "0,1"                                   | -                                    |
| `--model_name_or_path`         | Model to evaluate                                 | "Qwen/Qwen2.5-Omni-7B"                           | See supported models above           |
| `--context_minutes`            | Context window size in minutes                    | 120                                         | -                                    |
| `--data_source_path`           | Path to dialogue data                             | "train_test_data_split/dialogue_names.json" | -                                    |
| `--modals`                     | Input modalities                                  | ["text"]                                    | "text", "audio", or both             |
| `--is_text_with_audio_emotion` | Include audio emotion annotations for text models | False                                       | True/False                           |
| `--save_base_dir`              | Directory to save results                         | "results2"                                  | -                                    |
| `--audio_store`                | Audio input format                                | "base64_string"                             | "local_file", "url", "base64_string" |
| `--force_update`               | Force update existing results                     | False                                       | True/False                           |

### Examples

1. Evaluate a text-only model without audio annotations:
```bash
python evaluate.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --modals text \
    --data_source_path train_test_data_split/dialogue_names.json
```

2. Evaluate a text-only model with audio emotion annotations:
```bash
python evaluate.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --modals text \
    --is_text_with_audio_emotion \
    --data_source_path train_test_data_split/dialogue_names.json
```

3. Evaluate a multimodal model with local audio files:
```bash
python evaluate.py \
    --model_name_or_path Qwen/Qwen2.5-Omni-7B \
    --modals text audio \
    --audio_store local_file \
    --data_source_path train_test_data_split/dialogue_names.json
```

## Results Performance Compute

Evaluation results will be saved in the directory specified by `--save_base_dir`. 
The output includes parsed Model responses. Execute the following command to retrieve performance metrics for all eval setting.

```bash
python eval_performance_compute.py
```

## Training
We implement parameter-efficient fine-tuning via LoRA, validated through stratified 5-fold cross-validation.
Create a multi-configuration experiment script first, then execute its commands.
```bash
cd training
python train_sh_gen.py
```
Run the target command in generated train.sh to obtain your desired training results.
## License

Here's the *Rights and Permissions* section that clearly distinguishes between the openly available transcripts (non-commercial use) and restricted-access audio files:

### **Rights and Permissions**  
**Anonymized wav and Text transcripts** of the audio materials are made publicly available under a [**Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC)**](https://creativecommons.org/licenses/by-nc/4.0/). Users may share and adapt the transcripts for non-commercial purposes, provided proper attribution is given to the original authors and source. Commercial use requires separate permission from the copyright holders.
All publications using the audio data must cite this original work.

## Citation
If you use this code in your research, please cite:
[Your citation information here]