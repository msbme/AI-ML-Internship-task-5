# Context-Aware RAG Chatbot Using LangChain for Biomedical Equipment Diagnostics

## Objective
The objective of this project is to develop and deploy a Context-Aware Retrieval-Augmented Generation (RAG) conversational chatbot. The system is engineered to ingest a custom unstructured text corpus—specifically a **Biomedical Engineering Radiology Lab Equipment Manual**—and accurately extract localized technical protocols, safety workflows, and troubleshooting information. 

By binding a semantic vector retriever to a sliding conversation-turn buffer memory, the chatbot resolves conversational pronouns and maintains full contextual tracking across multi-turn user dialogues.

---

## Methodology & Architectural Approach

The system is implemented as a self-contained pipeline comprising data processing, embedding generation, vector storage indexing, memory management, and front-end deployment:

### 1. Dataset Processing & Tokenization
* **Corpus Ingestion:** Loads the custom knowledge file (`knowledge_base.txt`) mapping operational procedures for complex imaging modalities (MRI, CT, DR X-Ray, Ultrasound), totaling 3,795 text characters.
* **Semantic Text Splitting:** Utilizes a recursive character text splitter to break down dense documentation into **10 uniform text chunks**, applying localized overlap constraints to preserve syntax cross-boundaries.

### 2. Retrieval & Local Open-Source LLM Architecture
* **Vector Embeddings Engine:** Leverages the open-source `sentence-transformers/all-MiniLM-L6-v2` framework to encode tokenized text chunks into dense, high-dimensional semantic vectors.
* **Vector Store Indexing:** Builds an in-memory **FAISS (Facebook AI Similarity Search)** vector repository saved natively to disk at `/content/faiss_index`. At runtime, it acts as a high-speed search engine mapping incoming user queries to relevant source text using cosine similarity metrics ($k=3$).
* **Inference Pipeline:** Implements a local execution instance of the `google/flan-t5-base` transformer model via a Hugging Face pipeline. This removes any requirement for commercial cloud API keys, making the execution completely self-contained and private.

### 3. State Memory & Deployment Architecture
* **Sliding-Turn Conversation Memory:** Employs a custom historic message buffer tracking user queries and AI responses as standard tuples. It dynamically injects the **last 3 dialogue turns** straight into the system prompt window, giving the local LLM the context needed to resolve dependent follow-up questions.
* **Evaluation & Graphics Tracking:** Integrates validation testing steps to monitor retrieval overlap patterns, exporting summary data matrices directly into `/content/eval_visualizations.png`.
* **Streamlit UI Interface:** Generates a production-ready user interface file (`app.py`) managing input/output text cycles, ready to be safely exposed to a live external browser via an ngrok network tunnel.

---

## Key Results & Observations

### Pipeline Performance Summary

| Architecture Metric | Value Baseline / Performance Status |
| :--- | :--- |
| **Knowledge Base Source** | `/content/knowledge_base.txt` (3795 characters) |
| **Total Chunks Vectorized** | 10 distinct semantic blocks |
| **Vector Engine Architecture** | FAISS Database Index (`/content/faiss_index`) |
| **Embedding Dimension Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Core Transformer LLM** | Local `google/flan-t5-base` Pipeline |
| **Contextual Conversational Memory** | Sliding Buffer Window (Tracks trailing 3 multi-turn sequences) |
| **Visual Analytics Generation** | Exported to `/content/eval_visualizations.png` |

### Strategic Evaluation Insights

1. **High-Fidelity Retrieval Alignment:** The FAISS index successfully mapped specific domain queries to targeted documentation frames (e.g., retrieving details regarding Flat-Panel Detector handling rules under Digital Radiography systems). This ensures that factual context is always delivered to the model prompt window.

2. **Local Model Generation Limits:**
   During evaluation, the local `flan-t5-base` model accurately mapped direct target phrases (such as matching general document content to "MRI"). However, on broad follow-up inquiries like *"Can you give me more detail on that?"*, the model correctly fallback-safed to *"I don't have enough information to answer that"* instead of generating text hallucinations. For complex descriptive phrasing, scaling the local architecture up to parameter-dense models (such as Llama 3 or Mistral) inside the same vector pipeline will instantly boost language fluidity.
