# Financial-Analysis-Assistant
COMS-4995-032: Applied Machine Learning
Final Project

## Overview
The **Financial Analysis Assistant** is a retrieval augmented generation (RAG) engine designed to answer high-level financial questions about major semiconductor companies by utilizing company filings and recent news articles. 

The solution consists of the following:
- Retrieval of relevant vectorized chunks over parsed 10K filings and Q3 2025 quarterly reports
- Real-time recent news extraction
- Template-guided prompt creation
- LLM fine-tuning to get optimal responses for user query
- Citation-driven final responses

Documentation from the following semiconductor companies were used in this project:
- NVIDIA
- Intel
- TSMC (Taiwan Semiconductor Manufacturing Company)
- Samsung Electronics

## Business Problem
Financial analysis of semiconductor companies is fragmented across unstructured news articles, subjective op-eds, and dense filings. Analysts struggle to quickly extract relevant information and transform them to impactful insights without manually reading through dense documents.

Therefore, this RAG solution answers company-specific questions by combining financial filings (10k and quarterly reports), curated news, and structured response templates to produce explainable responses to order to enable faster and more reliable analysis.

## Solution Architecture Diagram
(add once finalized)

## Repository Structure

```text
Financial-Analysis-Assistant/
├── data/
│   ├── raw/                         # Original PDFs and source documents
│   ├── parsed/                      # Parsed JSON/text files
│   └── artifacts/                   # FAISS index, chunks, and chunk metadata
├── src/
│   ├── ingestion/                   # PDF parsing
│   ├── chunking/                    # Chunking logic (Embedding Model + Vector DB)
│   ├── finetuning-llm/              # LLM fine-tuning approach
│   └── app/
│       ├── app.py                   # Chat UI for end-user interaction
│       ├── template_embeddings.py   # Build the final prompt (financial chunks + news + selected prompt template)
│       └── prompt_builder.py        # LLM output (after fine-tuning)
├── docs/
│   ├── architecture/
│   ├── slides/
│   ├── demo/
│   └── report/
├── README.md
└── requirements.txt
```
## Chunking, Embeddings, and Vector Retrieval

This project includes a document chunking and vector indexing pipeline that prepares financial filings for retrieval-augmented generation (RAG).
The pipeline processes parsed financial documents, generates dense embeddings, builds a FAISS index, and enables semantic search over the indexed content.

### Input Data
Parsed documents should be uploaded.

Each file is expected to be a JSON file with the following structure:
```json
{
  "filename": "example.pdf",
  "pages": [
    { "page_number": 1, "text": "page text here" },
    { "page_number": 2, "text": "page text here" }
  ]
}
```
### Build Document Chunks, Embeddings, and FAISS Index
Run the next code:
```bash
python src/Build_Search_Embeddings.py
```
