# IIITG RAG Application

A Retrieval-Augmented Generation (RAG) application built to answer questions related to IIIT Gwalior using Large Language Models (LLMs), vector search, and semantic retrieval.

## Overview

This project enables users to interact with IIIT Guwahati-related information through a conversational interface. Instead of relying solely on the LLM's pre-trained knowledge, the system retrieves relevant information from a curated knowledge base and generates accurate, context-aware responses.

The application follows the Retrieval-Augmented Generation (RAG) paradigm, which combines document retrieval with generative AI to improve factual accuracy and reduce hallucinations.

---

## Features

- Document-based Question Answering
- Semantic Search using Vector Embeddings
- LLM-powered Response Generation
- Interactive Chat Interface
- Fast Retrieval with Vector Database
- Context-Aware Responses
- Support for PDF and Text Documents
- Retrieval-Augmented Generation Pipeline

---

## System Architecture

```text
User Query
    │
    ▼
Embedding Model
    │
    ▼
Vector Database
    │
Relevant Chunks Retrieved
    │
    ▼
LLM + Retrieved Context
    │
    ▼
Generated Response
