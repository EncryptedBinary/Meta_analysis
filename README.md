# Empowering Meta-Analysis: Leveraging Large Language Models for Scientific Synthesis

<p align="center">
  <strong>Official Implementation for IEEE BigData 2024 Submission</strong>
</p>

## Abstract

This study investigates automating meta-analysis in scientific documents using large language models (LLMs). Meta-analysis synthesizes findings from multiple studies but is labor-intensive when done manually. Our approach fine-tunes LLMs for efficient, automated meta-analysis using **Retrieval Augmented Generation (RAG)** and a novel loss metric, **Inverse Cosine Distance (ICD)**. The fine-tuned models achieved **87.6% relevant meta-analysis abstracts** and reduced irrelevance from **4.56% to 1.9%**, demonstrating efficiency in a low-resource environment.

[Here is the available Meta-Analysis Dataset (MAD)](Dataset)

---

### Dataset Statistics

| Metric                          | Actual | Chunked  |
|----------------------------------|--------|----------|
| **Types of Domains** :              Scientific Studies  
| **Min. input (Sj) context length** | 733    | 1005     |
| **Max. input (Sj) context length** | 32,767 | 2,000    |
| **Avg. input (Sj) context length** | 16,890.22 | 1,542.32 |
| **Min. labels (yj) context length** | 104    | 104      |
| **Max. labels (yj) context length** | 2,492  | 2,492    |
| **Avg. labels (yj) context length** | 1,446.45 | 1,446.45 |
| **Total Instances**              | 625    | 7,447    |

<table>
  <tr><th>Human Evaluators Details</th><th>Count</th></tr>
  <tr><td>Total no. of evaluators</td><td>13</td></tr>
  <tr><td>No. of female evaluators</td><td>4</td></tr>
  <tr><td>No. of male evaluators</td><td>9</td></tr>
  <tr><td>Avg. age</td><td>23</td></tr>
  <tr><td>Profession</td><td>Student, Engineer</td></tr>
</table>

---

## Methodology

1. **Data Preprocessing:** Chunking datasets for LLM context length restrictions.
2. **Fine-Tuning:** LLMs fine-tuned using **Inverse Cosine Distance (ICD)** to maximize relevance.
3. **RAG Integration:** Semantic search matches context with relevant data chunks for summary generation.
4. **Evaluation:** Human evaluators assess abstract readability and relevance.

![image](https://github.com/user-attachments/assets/fcdad47b-a932-425f-956c-e68b4198ee78)

<strong>Fig. 1: (a) Paraphraser-based approach [1], [2]; (b) Retrieval Augmentation Generation [3], [4]; (c) Our novel approach with fine-tuned LLMs.</strong>

## Result
#### Model Performance on Summarization Quality Across Benchmark Datasets

This table compares model performance on benchmark datasets for summarization quality without fine-tuning, enabling assessment across varying context lengths.

| **Method**     | **Models**                   | **Open-i (BLEU ↑)** | **Open-i (ROUGE ↑)** | **writer_summaries (BLEU ↑)** | **writer_summaries (ROUGE ↑)** | **CL-SciSumm (BLEU ↑)** | **CL-SciSumm (ROUGE ↑)** |
|----------------|------------------------------|---------------------|----------------------|-------------------------------|--------------------------------|-------------------------|--------------------------|
| **Established**|                              |                     |                      |                               |                                |                         |                          |
|                | GPT-4 with ICL          | 46.0                | 68.2                 | -                             | -                              | -                       | -                        |
|                | InstructGPT davinci v2    | -                   | -                    | -                             | -                              | 48                      | -                        |
|                | GCN Hybrid                | -                   | -                    | -                             | -                              | -                       | 33.88                    |
| **Context length restricted LLMs** |              |                     |                      |                               |                                |                         |                          |
| **Pre-trained**| Falcon 7B                 | 0.19                | 3.17                 | 0.76                          | 5.19                           | 0.71                    | 2.21                     |
| **Pre-trained**| Gemma 7B                  | 2.13                | 8.81                 | 4.47                          | 30.28                          | 2.44                    | 20.78                    |
| **Pre-trained**| Orca-2 7B                 | 3.53                | 8.36                 | 4.29                          | 22.51                          | 2.86                    | 15.55                    |
| **Pre-trained**| StableLM-Base-Alpha 7B    | 2.01                | 2.45                 | 3.56                          | 15.36                          | 1.17                    | 16.58                    |
| **Pre-trained**| Llama-2 7B                | 4.81                | 10.28                | 5.21                          | 31.61                          | 3.01                    | 22.84                    |
| **Pre-trained**| Mistral-v0.1 7B           | 1.21                | 6.57                 | 1.62                          | 6.37                           | 0.36                    | 2.55                     |
| **Ours**       | Llama-2 7B FT                 | **10.14**           | **27.39**            | **12.66**                     | **31.36**                      | 7.15                    | 25.22                    |
| **Ours**       | Mistral-v0.1 7B FT            | **12.42**           | **31.57**            | **14.56**                     | **35.56**                      | **8.38**                | **27.29**                |

#### Notes

1. **Open-i:** Medical radiological dataset. Generated summaries from 100 samples.
2. **writer_summaries:** Article summarization dataset, evaluated on 120 samples.
3. **CL-SciSumm:** Large corpus containing scientific article data, evaluated on 20 samples. Chunking required due to context length limitations.
4. **Established:** Pre-established methods from the cited papers for the three specific datasets. BLEU and ROUGE scores are not comparable with the other models due to different evaluation methodologies.


## 🚀 Quick Start

To get started with our models, follow the steps below.

### 1. Clone the Repository
```bash
git clone https://github.com/EncryptedBinary/Meta_analysis.git
cd meta-analysis-llm
```
2. Install Required Packages
Run the following commands to install the necessary libraries:
```python 
!pip install transformers trl accelerate torch bitsandbytes peft datasets -qU
!pip install langchain
```

3. Using Our Pre-trained Models
You can use our pre-trained models for generating meta-analysis abstracts by downloading them from Hugging Face:
```bash
Llama2: Bakugo123/Cosine_matric_llama2_prompt1
Mistral: bingowithmylingo/mistral_newPrompt

```
Simply load the models and run inference using the fine-tuned weights. Example below:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Bakugo123/Cosine_matric_llama2_prompt1")
tokenizer = AutoTokenizer.from_pretrained("Bakugo123/Cosine_matric_llama2_prompt1")

# Example inference
input_text = "Given a collection of abstracts from papers used in various medical fields for
meta-analysis, generate a meta-analysis abstract. Summarize the key findings and
provide numerical values or statistical information for specific observations that
are commonly reported in the provided abstracts."
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs)
print(tokenizer.decode(output, skip_special_tokens=True))
```
## 🧪 Train-Test-Split

- **Training**: 400 meta-analysis documents
- **Validation**: 75 meta-analysis documents
- **Testing**: 50 meta-analysis documents

Feel free to modify the splits or experiment with different datasets based on your use case.

## 📚 Model Training

For those interested in fine-tuning the models further, we recommend checking out the `train.py` script, which includes hyperparameters and configurations for:

- **Epochs**: 10
- **Loss Function**: Inverse Cosine Distance (ICD)
- **Optimization**: Using bitsandbytes for efficient scaling


### References
- [1] M. Subbiah et al., "Reading subtext: Evaluating large language models," arXiv:2403.01061, 2024.
- [2] J. Lim, H.-J. Song, "Improving multi-stage long document summarization," 2023.
- [3] A. J. Yepes et al., "Financial report chunking for effective retrieval," arXiv:2402.05131, 2024.
- [4] S. Manathunga, Y. Illangasekara, "Retrieval augmented generation in medical education," arXiv:2308.00479, 2023.

---

