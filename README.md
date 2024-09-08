# Empowering Meta-Analysis: Leveraging Large Language Models for Scientific Synthesis

<p align="center">
  <strong>Official Implementation for IEEE BigData 2024 Submission</strong>
</p>

## Abstract

This study investigates the automation of meta-analysis in scientific documents using large language models (LLMs). Meta-analysis is a robust statistical method that synthesizes the findings of multiple studies (support articles) to provide a comprehensive understanding. We know that a meta-article provides a structured analysis of several articles. However, conducting meta-analysis by hand is labor-intensive, time-consuming, and susceptible to human error, highlighting the need for automated pipelines to streamline the process.

Our research introduces a novel approach that fine-tunes the LLM on extensive scientific datasets to address challenges in big data handling and structured data extraction. We automate and optimize the meta-analysis process by integrating **Retrieval Augmented Generation (RAG)**. Tailored through prompt engineering and a new loss metric, **Inverse Cosine Distance (ICD)**, designed for fine-tuning on large contextual datasets, LLMs efficiently generate structured meta-analysis content. Human evaluation then assesses relevance and provides information on model performance in key metrics.

This research demonstrates that fine-tuned models outperform non-fine-tuned models, with fine-tuned LLMs generating **87.6% relevant meta-analysis abstracts**. The relevance of the context, based on human evaluation, shows a reduction in irrelevancy from 4.56% to 1.9%. These experiments were conducted in a low-resource environment, highlighting the study’s contribution to enhancing the efficiency and reliability of meta-analysis automation.

[Here is the available dataset.](https://github.com/YourRepoLinkHere)

---

## Dataset Statistics

<table>
  <caption><strong>TABLE I: Detailed statistics of the actual dataset and demographics of human evaluators</strong></caption>
  <tr>
    <th>Dataset Statistics</th>
    <th>Actual</th>
    <th>Chunked</th>
  </tr>
  <tr>
    <td>Types of Domains</td>
    <td colspan="2">Scientific Studies</td>
  </tr>
  <tr>
    <td>Min. input (Sj) context length</td>
    <td>733</td>
    <td>1005</td>
  </tr>
  <tr>
    <td>Max. input (Sj) context length</td>
    <td>32,767</td>
    <td>2000</td>
  </tr>
  <tr>
    <td>Avg. input (Sj) context length</td>
    <td>16,890.22</td>
    <td>1,542.32</td>
  </tr>
  <tr>
    <td>Min. labels (yj) context length</td>
    <td colspan="2">104</td>
  </tr>
  <tr>
    <td>Max. labels (yj) context length</td>
    <td colspan="2">2,492</td>
  </tr>
  <tr>
    <td>Avg. labels (yj) context length</td>
    <td colspan="2">1,446.45</td>
  </tr>
  <tr>
    <td>Total Instances</td>
    <td>625</td>
    <td>7,447</td>
  </tr>
</table>

<table>
  <tr>
    <th>Human Evaluators Details</th>
    <th>Count</th>
  </tr>
  <tr>
    <td>Total no. of evaluators</td>
    <td>13</td>
  </tr>
  <tr>
    <td>No. of female evaluators</td>
    <td>4</td>
  </tr>
  <tr>
    <td>No. of male evaluators</td>
    <td>9</td>
  </tr>
  <tr>
    <td>Avg. age</td>
    <td>23</td>
  </tr>
  <tr>
    <td>Profession</td>
    <td>Student, Engineer</td>
  </tr>
  <tr>
    <td>Education Background</td>
    <td>Undergraduate</td>
  </tr>
</table>

---

## Methodology

The approach used in this paper involves multiple techniques to handle the unique challenges posed by meta-analysis automation. Specifically, we integrate fine-tuned LLMs with context optimization techniques. Below is a general overview of the methods:

1. **Data Preprocessing:** The dataset is chunked due to context length restrictions imposed by LLMs. We prioritize small LLMs over resource-intensive large models.
2. **Model Fine-Tuning:** LLMs are fine-tuned using a new loss metric, **Inverse Cosine Distance (ICD)**, designed to maximize relevance and minimize irrelevance in generated abstracts.
3. **Retrieval Augmented Generation (RAG):** Semantic search is employed to match input context with relevant chunks before passing it to the fine-tuned model to generate abstracts.
4. **Evaluation:** Human evaluators assess the readability and relevance of the generated abstracts.

<p align="center">
  <img src="![image](https://github.com/user-attachments/assets/00603aed-a377-4a8d-af36-7acbfb02ca5c)
" alt="Methodology Overview" width="600px">
</p>
<p align="center">
  <strong>Fig. 1: (a) Paraphraser-based approach that combines multiple generated summary chunks from LLMs has been used by [16], [17], (b) Retrieval augmentation generation-based approach has been applied in [18], [19] using a vector database to store chunked data and cluster them before passing to LLM to produce a summary. Existing methods often fall short of handling big scientific contextual data and generating structured synthesis. (c) We propose a novel approach involving fine-tuning LLMs with large contexts and utilizing them to generate meta-analysis abstracts. Abstracts from support papers serve as input, with meta-papers’ abstracts as labels. Pre-processing involves chunking the dataset due to context length restrictions and prioritizing small LLMs over resource-intensive large LLMs. The fine-tuned model generates meta-analysis abstracts via semantic search with the provided context and query.</strong>
</p>

### References
- [16] M. Subbiah, S. Zhang, L. B. Chilton, and K. McKeown, “Reading subtext: Evaluating large language models on short story summarization with writers,” arXiv preprint arXiv:2403.01061, 2024.
- [17] J. Lim and H.-J. Song, “Improving multi-stage long document summarization with enhanced coarse summarizer,” in Proceedings of the 4th New Frontiers in Summarization Workshop, 2023, pp. 135–144.
- [18] A. J. Yepes, Y. You, J. Milczek, S. Laverde, and L. Li, “Financial report chunking for effective retrieval augmented generation,” arXiv preprint arXiv:2402.05131, 2024.
- [19] S. Manathunga and Y. Illangasekara, “Retrieval augmented generation and representative vector summarization for large unstructured textual data in medical education,” arXiv preprint arXiv:2308.00479, 2023.

---

## License
This repository is licensed under the MIT License. See the `LICENSE` file for more information.

