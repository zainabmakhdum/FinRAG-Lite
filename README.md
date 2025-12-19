# FinRAG-Lite
COMS-4995-032: Applied Machine Learning Final Project

**FinRAG-Lite** is a retrieval augmented generation (RAG) engine designed to answer high-level financial questions about major semiconductor companies by utilizing company filings and recent news articles. 

# Contributions: - /docs/Logistics_Planning

## 95% - Brandon Rodriguez, Zainab Makhdum 

### Brandon: 
Researched and implemented local quantized LLM model from HuggingFace. Developed answer template categories, usage definitions, and patterns for the LLM to use. Developed final prompt template to optimize performance, accuracy, and stability. Built testing procedure to evaluate model by comparing 10 queries using 3 LLM models and a judge LLM. Generated 250 queries and 100 synthetic examples using prompt wrapper functions and manual LLM prompting. Researched and implemented QLoRA fine-tuning using generated examples. Built final UI and connected embedding, prompt creation, and LLM parsing functions. Wrote technical details to slideshow and final report. Organized final notebook files into py files for Github. 

### Zainab: 
Researched and implemented a live news API for news embedding using token authentication and keyword searches. Planned and assigned all work each week, following up on any blockers or missing pieces. Created final prompt maker notebook used to generate all prompts used in tuning and testing. Planned and built slideshow including technical slides to manage 10 minute timer. Wrote video script according to time constraints to ensure equal participation. Created shared Github repo and file structure. Helped write and organize final report. 

## 5% - Nazgul Maksutkhan, Mani Karunanidhi, and Ayon Roy

### Nazgul: 
Researched and implemented a lightweight embedded vector system to pull relevant snippets when creating a prompt using a FAISS index. Built prompts to evaluate model performance and executed testing steps. 

### Mani: 
Built a chunking system to divide company documents into usable snippets with metadata. Chunked collected company filings into Json files. Researched initial UI app format. Tabulated testing results.

### Ayon: 
Researched Google Cloud TPU strength and cloud storage size. Researched initial UI app format. 

![alt text](/images/appimage.png)

# Requirements and Instructions

## Requirements

### 1) Apple computer with an M-series chip (required)
This project uses **MLX / `mlx-lm`** for local LLM inference, which currently requires **Apple Silicon (M1 / M2 / M3 / M4)**. The app is intended to be run locally on a Mac with an M-chip.

### 2) Install dependencies from `requirements.txt`
The `requirements.txt` file is located in **`src/`** (same folder as `app.py`).

From the repo root:
```bash
pip install -r src/requirements.txt
```

> Note (FAISS on Apple Silicon): if you run into issues installing FAISS via pip, install it via conda-forge:
```bash
conda install -c conda-forge faiss-cpu
```

### 3) Download the Hugging Face model (recommended)
The fine-tuned / merged MLX model is hosted publicly at:

- **Hugging Face repo:** `br2835/mistraladaptmerged`  
- **Download size:** ~11 GB

You can either:
- **Pre-download it** (recommended), or
- Let the app download it automatically on first run (slower first launch).

## Optional: Optimize M-chip performance (recommended)
For better stability/performance on larger MLX workloads, you can increase the Metal “wired” memory limit:

```bash
sudo sysctl iogpu.wired_limit_mb=<MB_VALUE>
```

Recommendation:
- Set `<MB_VALUE>` to approximately: **`total_vram_mb - 4500`**
- Example: if your total VRAM is ~16384 MB, you might set:
  `iogpu.wired_limit_mb=11884`

(You can adjust slightly up/down depending on your system stability.)

---

## Instructions

### Run the app (terminal)
From the repo root:
```bash
python src/app.py
```

Then open the local URL printed in the terminal (Gradio will show something like `http://127.0.0.1:7860`).

### Run the app (Jupyter notebook)
In a notebook cell:
```python
%run src/app.py
```

### Time estimate
On an **M1 Mac with ~16GB VRAM**, a single query typically takes **~180–200 seconds** end-to-end (prompt build + local inference + formatting).  
If you have a stronger M-chip / more memory, performance will generally improve.

## Project Overview and Sections
### Overview
The solution consisted of the following:
- Retrieval of relevant vectorized chunks over parsed 10K filings and Q3 2025 quarterly reports
- Real-time recent news extraction API configuration
- Template-guided prompt creation
- LLM fine-tuning to get optimal responses for user query
- Citation-driven final responses
- Interactive UI app

Documentation from the following semiconductor companies were used in this project:
- NVIDIA
- Intel
- TSMC (Taiwan Semiconductor Manufacturing Company)
- Samsung Electronics

### Business Problem - /docs/Logistics_Planning
Financial analysis of semiconductor companies is fragmented across unstructured news articles, subjective op-eds, and dense filings. Analysts struggle to quickly extract relevant information and transform them to impactful insights without manually reading through dense documents.

Therefore, this RAG solution answers company-specific questions by combining financial filings (10k and quarterly reports), curated news, and structured response templates to produce explainable responses to order to enable faster and more reliable analysis.

### Chunking, Embeddings, and Vector Retrieval - /docs/Chunking and docs/EmbeddedPrompt

We chunked each parsed filing/report into retrieval-sized passages, stored the chunk text plus metadata (e.g., source filename and page number), and embedded those chunks into dense vectors for semantic search. At runtime, the user query is embedded and used to retrieve the most relevant evidence (top snippets) from a FAISS index, providing grounded context for the RAG answer generation. The final embedding workflow is designed to return a small, consistent context window (snippets/templates/news) for prompt construction.

### Prompt Engineering - docs/EmbeddedPrompt

We engineered a structured prompt format that enforces evidence-only reasoning (snippets as the source of truth), uses news only as supplementary context, and requires explicit, snippet-based citations to reduce hallucinations and improve interpretability. A library of 28 templates forces the model to pick a single response pattern and “slot-fill” it, creating consistent outputs across bullish/bearish, risk, peer comparison, and limited-information cases. The prompt also specifies a strict single-line JSON output schema to make downstream parsing reliable and deterministic. 

### Local Quantized Large Language Model and QLoRA Fine Tuning - /docs/FineTuning

We used a 24B instruction-tuned model and applied QLoRA fine-tuning (4-bit NF4 base quantization + LoRA adapters) using a distilled synthetic dataset generated from a stronger teacher model (ChatGPT-5.2 Thinking), iterating over multiple training rounds to reduce overfitting and improve generalization. After training, we merged the adapters into the full model, uploaded the merged checkpoint to Hugging Face (br2835/mistraladaptmerged), and produced an MLX-friendly 3-bit quantized variant so the system can run locally on Apple Silicon. This approach preserves strong instruction-following while making local inference feasible on consumer hardware (e.g., an M1 laptop).

### UI App and Class Creation - /src

We modularized the pipeline into importable components so the UI remains a thin orchestrator: prompt construction and retrieval live in PromptMaker.py, model execution + output parsing live in LLMCalls.py, and app.py wires everything together into an interactive Gradio interface. The UI takes a user question, builds a template-guided prompt with retrieved snippets and optional news, runs the local model, and then transforms the raw JSON output into a final response with clickable document citations plus a short evidence/news summary. This design makes the system easier to test, extend, and reuse while keeping the app logic clean and maintainable.

### Repository Structure

```text
FinRAG-Lite/
├── README.md
├── data/
│   ├── ParsedCompanyFilings/
│   │   └── ZippedChunkedCompanyFiles.zip
│   ├── artifacts/
│   │   ├── chunk_embeddings.npy
│   │   ├── chunk_meta.json
│   │   ├── chunks.json
│   │   └── faiss.index
│   └── data.py
├── docs/
│   ├── Chunking/
│   ├── EmbeddedPrompt/
│   │   ├── Final Prompt Example.pdf
│   │   ├── FinalEmbeddingPromptNotebook.ipynb
│   │   └── Template List.pdf
│   ├── Logistics_Planning/
│   │   └── Project Ideas and Datasets.pdf
│   ├── QLoRAFineTuning/
│   │   ├── Brandon LLM Research.pdf
│   │   ├── FineTuningCode.ipynb
│   │   ├── LoRAGenCode.ipynb
│   │   ├── Training Queries.pdf
│   │   ├── all_aml_full.jsonl
│   │   ├── aml_tuning_dataset_with_prompts.xlsx
│   │   ├── mergecoder.ipynb
│   │   ├── train_aml.jsonl
│   │   └── valid_aml.jsonl
│   ├── Testing/
│   │   └── aml_test_dataset_template.xlsx
│   └── docs.py
└── src/
    ├── Build_Search_Embeddings.py
    ├── LLMCalls.py
    ├── PromptMaker.py
    ├── app.py
    ├── embedding_data/
    │   ├── chunk_embeddings.npy
    │   ├── chunk_meta.json
    │   ├── chunks.json
    │   └── faiss.index
    └── src.py
```
