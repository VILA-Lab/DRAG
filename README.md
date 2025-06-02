# DRAG

Code for the paper "DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination via Evidence and Graph-based Distillation (ACL 2025 Main)."

## Table of Contents

- [Motivation](#motivation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contribution](#contribution)
- [Citation](#citation)

---

## Motivation

RAG methods have proven effective for tasks requiring factual consistency and robust knowledge retrieval. However, large-scale RAG systems are prone to generating "hallucinated" content. This repo provides the code to run DRAG, a novel framework for distilling RAG knowledge from large-scale Language Models (LLMs) into small LMs (SLMs). Our approach leverages evidence- and knowledge graph–based distillation, ensuring that the distilled model retains critical factual knowledge while significantly reducing model size and computational cost. By aligning the smaller model's predictions with a structured knowledge graph and ranked evidence, DRAG effectively mitigates hallucinations and improves factual accuracy. Experimental evaluations on multiple benchmarks demonstrate that our method outperforms the prior competitive RAG methods like MiniRAG for SLMs by up to 27.7% using the same models, preserving high-level efficiency and reliability.


## Usage

1. Create a .env file containing the private keys for all the LLMs that will be utilzied for evidence and graph generation.

```bash
GROQ_KEY='abc'
OPENAI_KEY='def'
GEMINI_KEY='ghi'
CLAUDE_KEY='jkl'
```

2. Change the following parameters in the `language_model.py` file:
- The desired model names for each LLM (modify the class definitions)
- The MAX_RETRIES variable based on the desired number of max retries for calling APIs

3. Run the following command in the terminal to execute the graph and evidence generation pipeline:
```bash
python 0_generate_all_context.py <llm-provider> <benchmark> <num-to-generate> [options]
```

| Argument / Option    | Description                                                                   |
| -------------------- | ----------------------------------------------------------------------------- |
| `<llm-provider>`     | **(Required)** Name of the large LLM to be used for evidence/graph generation |
| `<benchmark>`        | **(Required)** Name of benchmark used for evaluation                          |
| `<num-to-generate>`  | **(Required)** Number of evidences and graph relationships to generate        |
| `--multithread`      | Enable multithreading                                                         |

Supported `llm-provider` options:
- [gpt](https://openai.com/api/)
- [gemini](https://ai.google.dev/)
- [claude](https://www.anthropic.com/api)
- [llama](https://www.llama.com/products/llama-api/)
- [deepseek](https://api-docs.deepseek.com/)

Supported `benchmark` options:
- [mmlu](https://huggingface.co/datasets/cais/mmlu)
- [fever](https://huggingface.co/chenxwh/AVeriTeC)
- [med](https://huggingface.co/datasets/openlifescienceai/medmcqa)
- [open](https://huggingface.co/datasets/Open-Style/Open-LLM-Benchmark)
- [gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)
- [web](https://huggingface.co/datasets/stanfordnlp/web_questions)
- [searchqa](https://huggingface.co/datasets/lucadiliello/searchqa)

4. Verify that the output files contain the generated evidences/graphs:
- `evidences_{llm-provider}_{benchmark}.csv` (source code `1_generate_evidences.py`): Contains the output evidences for each question in the specified benchmark
- `evidences_final_{llm-provider}_{benchmark}.csv` (source code `2_generate_evidence_rankings.py`): Contains the evidences with their relevance order based on LLM ranking, semantic ranking, and combined (LLM + semantic) ranking
- `graph_{llm-provider}_{benchmark}.csv` (source code `3_generate_graph.py`): Generates graph relationships for each question in the specified benchamrk using the previously generated evidences
- `graph_final_{llm-provider}_{benchmark}.csv` (source code `4_generate_graph_rankings.py`): Contains graph relationships with their relevance order based on LLM ranking, semantic ranking, and combined (LLM + semantic) ranking

5. Optionally, run `5_generate_responses_no_context.py` to generate the responses for the small LLM without evidence/graph context, and run `6_generate_responses.py` to generate the responses with evidence and/or graph context. Change the model versions in `language_model.py` before running these scripts. 

    NOTE: In our paper, we used [Harness](https://github.com/EleutherAI/lm-evaluation-harness) for response generation; this framework also provides evaluation. 


## Contribution
We welcome contributions - please feel open an issue, or a pull request, if you have any suggestions/improvements.

## Citation
