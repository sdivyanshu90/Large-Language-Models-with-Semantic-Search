# Large Language Models with Semantic Search

*Course by [DeepLearning.AI](https://www.deeplearning.ai/) in collaboration with [Cohere](https://cohere.com/)*

![LLMs](https://img.shields.io/badge/LLMs-Semantic%20Search-green)
![Cohere](https://img.shields.io/badge/Cohere-Partner-purple)

---

## 📌 Overview

This repository provides a detailed summary of the **"Large Language Models with Semantic Search"** course.

The course explores how **Large Language Models (LLMs)**, when combined with **semantic search techniques**, can go far beyond traditional keyword search.
You will learn how to:

* Represent text as embeddings,
* Retrieve relevant documents,
* Re-rank results for higher accuracy,
* And finally, generate meaningful answers.

---

## 🎯 Course Topics

### 1️⃣ Keyword Search

🔎 **Definition**:
A traditional search approach that matches query terms with words in documents.

⚙️ **How it works**:

* Breaks a query into tokens (keywords).
* Looks for exact matches in the document corpus.
* Ranks documents by keyword frequency (e.g., TF-IDF, BM25).

📌 **Strengths**:

* Simple, efficient, and interpretable.
* Works well when queries and documents share the same words.

⚠️ **Limitations**:

* Cannot capture **synonyms** or **semantic meaning**.
* Sensitive to spelling variations.
* Example: Searching "AI" won’t return "Artificial Intelligence" unless explicitly included.

💡 **Takeaway**: Keyword search is the foundation but limited for modern, meaning-driven queries.

---

### 2️⃣ Embeddings

🧩 **Definition**:
Embeddings are **vector representations** of text (words, sentences, or documents) that capture semantic meaning.

⚙️ **How it works**:

* Text is passed through a model (like word2vec, GloVe, or modern transformer-based models).
* Each word/sentence is mapped to a high-dimensional vector.
* Semantically similar texts end up close together in the vector space.

📌 **Key Properties**:

* Capture **contextual meaning**, not just surface forms.
* Enable comparison between texts using similarity metrics (cosine similarity, dot product).

📊 **Example**:

* "king" - "man" + "woman" ≈ "queen"
* "doctor" and "physician" will be close in vector space.

💡 **Takeaway**: Embeddings make it possible to search based on **meaning**, not just words.

---

### 3️⃣ Dense Retrieval

📖 **Definition**:
Dense retrieval uses **embeddings** to fetch relevant documents.

⚙️ **How it works**:

1. Encode the **query** into a vector.
2. Encode all **documents** into vectors.
3. Compute similarity between query and document vectors.
4. Return the documents with the highest similarity scores.

📌 **Benefits**:

* Captures semantic relationships.
* Works even if the query and document don’t share exact words.

📊 **Example**:
Query: *"How does a neural network learn?"*
Retrieved doc: *"Neural nets adjust their weights during training to minimize error."*

⚠️ **Challenges**:

* Requires efficient storage and search (ANN: Approximate Nearest Neighbors).
* May retrieve loosely relevant docs if embeddings are poor.

💡 **Takeaway**: Dense retrieval powers **semantic search** by matching concepts, not just terms.

---

### 4️⃣ Re-Rank

📖 **Definition**:
Re-ranking improves retrieval accuracy by **ordering results more intelligently**.

⚙️ **How it works**:

* Step 1: Dense retrieval fetches a candidate set (e.g., top 50 docs).
* Step 2: A stronger model (cross-encoder or LLM) re-evaluates each doc against the query.
* Step 3: Produces a new ranking, with the most relevant results on top.

📌 **Benefits**:

* Improves **precision** of search.
* Handles subtle differences in context.

📊 **Example**:
Query: *"Apple revenue 2023"*

* Initial retrieval: docs about fruit 🍎 and company 🍏.
* Re-rank: prioritizes docs about **Apple Inc. financial reports**.

💡 **Takeaway**: Re-ranking adds an extra layer of **intelligence** to search.

---

### 5️⃣ Generating Answers

📖 **Definition**:
Instead of returning documents, LLMs can **synthesize answers** from retrieved content.

⚙️ **How it works**:

1. Retrieve documents using dense retrieval.
2. Provide retrieved content as **context** to the LLM.
3. LLM generates a **natural language response**.

📌 **Benefits**:

* Provides **direct answers** rather than links.
* Handles summarization, reasoning, and rephrasing.

📊 **Example**:
Query: *"What is machine learning?"*

* Retrieval: Articles containing definitions.
* Generated Answer:
  *"Machine learning is a subset of artificial intelligence where algorithms learn patterns from data to make predictions without being explicitly programmed."*

💡 **Takeaway**: Answer generation is the final step — transforming retrieval into **user-friendly outputs**.

---

## 🔄 End-to-End Workflow

```mermaid
flowchart LR
    A[User Query] --> B[Convert to Embedding]
    B --> C[Dense Retrieval]
    C --> D[Candidate Documents]
    D --> E[Re-Rank with Model]
    E --> F[Top Relevant Docs]
    F --> G[Answer Generation with LLM]
    G --> H[Final Answer]
```

---

## 🙌 Acknowledgements

This learning material is based on the course:

📘 **[Large Language Models with Semantic Search](https://learn.deeplearning.ai/)**

* Provider: [DeepLearning.AI](https://www.deeplearning.ai/)
* Partner: [Cohere](https://cohere.com/)

Special thanks to **Andrew Ng** and the course contributors for creating this learning pathway.

---

## 📚 Resources

* [DeepLearning.AI Courses](https://learn.deeplearning.ai/)
* [Cohere Documentation](https://docs.cohere.com/)
* [Weaviate Semantic Search Guide](https://docs.weaviate.io/academy/py/starter_text_data/text_searches/semantic)
