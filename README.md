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
