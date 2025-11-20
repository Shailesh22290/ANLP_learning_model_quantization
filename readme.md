# Advanced NLP: Model Quantization for Efficient Emotion Classification

## Project Overview

This repository contains the implementation and evaluation for Assignment 2 of the Advanced Natural Language Processing (DSE 418/618) course. The core objective was to explore and compare various model compression and efficiency techniques—specifically quantization methods—to optimize a fine-tuned language model for the multi-class **Emotion Classification** task.

We compare the performance (Accuracy, F1-Score) and inference efficiency (time per sample) of four model variants:

1.  **Baseline Fine-Tuned Model:** A standard full-precision (FP16/FP32) fine-tuned Transformer.
2.  **Post-Training Quantization (PTQ):** Quantizing the Baseline model to 8-bit integers after fine-tuning.
3.  **Quantization-Aware Training (QAT):** Simulating quantization during the fine-tuning process to minimize performance degradation.
4.  **Quantized Low-Rank Adaptation (Q-LoRA):** Utilizing the Q-LoRA framework for both parameter-efficient fine-tuning and efficient deployment.

## Dataset

The project uses the **`dair-ai/emotion`** dataset, a multi-class text classification dataset comprising 6 common emotions.

## Prerequisites

The project relies on a GPU runtime (e.g., Google Colab GPU) and the following key libraries:

* `torch`
* `transformers`
* `datasets`
* `peft` (for LoRA implementation)
* `bitsandbytes` (for 4-bit and 8-bit operations)
* `accelerate`
* `seaborn` and `matplotlib` (for result visualization)

All dependencies are installed automatically within the provided Colab notebook.

## Repository Structure

* **`22290_shailesh_Advance nlpassignment2.ipynb`**: The primary Python (Colab) notebook containing all the necessary code for loading weights, running inference, evaluating performance, and generating the final comparison table.
* **`weights/`**: This conceptual directory holds the trained model weights for the Baseline, PTQ, QAT, and Q-LoRA models. *(These weights are loaded directly from a public URL in the notebook for reproducibility).*

## How to Reproduce Results (Inference & Evaluation)

The submission focuses on a reproducible evaluation pipeline. The full training code (fine-tuning, QAT, etc.) is included but commented out to ensure the notebook runs quickly and meets the time limit requirements.

### **Steps to Run `22290_shailesh_Advance nlpassignment2.ipynb`**

1.  **Access the Notebook:** Open the `22290_shailesh_Advance nlpassignment2.ipynb` file in Google Colab.
2.  **Select Runtime:** Go to `Runtime` -> `Change runtime type` and ensure **GPU** is selected.
3.  **Execute All:** Click `Runtime` -> `Run all`.



## Results

The quantitative performance comparison (Accuracy, Macro F1, and Inference Speed) for all four models is mentioned as a  end of the `22290_shailesh_Advance nlpassignment2.ipynb` execution.
