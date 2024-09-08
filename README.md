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
| **Types of Domains**              Scientific Studies  
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

### References
- [1] M. Subbiah et al., "Reading subtext: Evaluating large language models," arXiv:2403.01061, 2024.
- [2] J. Lim, H.-J. Song, "Improving multi-stage long document summarization," 2023.
- [3] A. J. Yepes et al., "Financial report chunking for effective retrieval," arXiv:2402.05131, 2024.
- [4] S. Manathunga, Y. Illangasekara, "Retrieval augmented generation in medical education," arXiv:2308.00479, 2023.

---

