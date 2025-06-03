# NLP Pipeline for Movie Reviews
Natural Language Processing on Movie Reviews: From Term Extraction to Knowledge Graphs and Deep Learning

This project presents a comprehensive Natural Language Processing (NLP) workflow for analyzing movie reviews. The work covers **text preprocessing, vectorization, clustering, sentiment analysis, topic modeling, knowledge graph construction, and deep learning classification (RNN/LSTM)**.

The focus is on real-world NLP problem-solving, leveraging both traditional and modern techniques, including **rule-based methods, transformer models, and neural architectures**.


---

## 📁 Repository Structure

| File/Folder                                | Description                                                    |
|-------------------------------------------|----------------------------------------------------------------|
| `Corpus_Final_Sec56_v1_20250412` | 230 movie reviews on 23 movies and 4 genres |
| `Part 1_Vectorization_and_Term_Analysis.ipynb` | Preprocessing, TF-IDF, Word2Vec, Doc2Vec, ELMo vectorization on movie reviews |
| `Part 2_Clustering_Classification_and_Sentiment_Analysis.ipynb` | Clustering (KMeans), sentiment analysis (TF-IDF, ML, BERT), genre classification (SVM, LR, etc.), topic modeling (LDA, BERTopic) |
| `Part 3_Knowledge_Graph_and_Deep_Learning_Report.ipynb` | Entity-relation extraction (spaCy, BERT, LLM), knowledge graph construction and visualization, RNN/LSTM genre classification |
| `Part 1_Vectorization_and_Term_Analysis_Report.pdf` | Detailed analysis and results of vectorization and term selection experiments |
| `Part 2_Clustering_Classification_and_Sentiment_Analysis_Report.pdf` | Clustering, classification, and sentiment analysis results, metrics, and findings |
| `Part 3_Knowledge_Graph_and_Deep_Learning_Report.pdf` | Final report on knowledge graph construction and RNN/LSTM genre classification models |

---

## 📊 Data Source

The primary dataset for this project is sourced from the belowing source:

(https://github.com/barrycforever/MSDS_453_NLP/tree/main/MSDS453_ClassCorpus)

This dataset consists of movie reviews used for various NLP experiments including vectorization, classification, and knowledge graph construction.

---

## 🔍 Project Highlights

- **Vectorization Methods:** Applied TF-IDF, Word2Vec, Doc2Vec, ELMo for document and token representation.
- **Clustering & Topic Modeling:** Performed KMeans clustering, LSA, LDA, BERTopic topic modeling; evaluated cluster quality and topic interpretability.
- **Sentiment & Genre Classification:** Implemented multiple classifiers (Logistic Regression, SVM, Random Forest, BERT) for sentiment and multi-class genre prediction.
- **Knowledge Graphs:** Extracted entities and relations using spaCy, BERT, and GPT-3.5-turbo; built and visualized graphs with NetworkX.
- **Deep Learning Models:** Trained RNN and LSTM models with varying architectures for genre prediction; compared results against traditional ML models.
- **Insights:** Explored the trade-off between rule-based, transformer-based, and deep learning approaches in NLP pipelines.

---

## 🧠 Key Learnings

- Effective text preprocessing and vectorization are critical for downstream NLP tasks.
- Transformer models (BERT, GPT) provide strong entity recognition but may require fine-tuning for relation extraction.
- Deep learning models (LSTM) outperform RNNs and classical models in complex text classification tasks.
- Knowledge graphs benefit from combining structured extraction with domain knowledge (e.g., equivalent classes for entities).

---

## 🌱 Future Directions

- Fine-tune LLMs for relation extraction and semantic search.
- Integrate knowledge graphs with retrieval-augmented generation (RAG) pipelines.
- Explore hybrid architectures combining symbolic AI (KGS) and neural models for enhanced text understanding.

---

## 📊 Figures and Results

Figures and visualizations are included in the reports (PDFs). Key graphs include:
- Knowledge Graph visualizations (spaCy-only, BERT, LLM+spaCy)
- Confusion matrices and accuracy/loss plots for RNN and LSTM models
- Cosine similarity heatmaps for vectorization methods

---

## 📚 References

- Core NLP libraries: spaCy, scikit-learn, TensorFlow/Keras, Hugging Face Transformers
- Research Reference: [Enhanced Transformer Architecture for NLP (ArXiv)](https://arxiv.org/abs/2310.10930)
- Additional reading: [The Transformer Architecture with Hybrid Models (Medium)](https://medium.com/@bijit211987/the-transformer-architecture-with-hybrid-models-eca885e12056)
