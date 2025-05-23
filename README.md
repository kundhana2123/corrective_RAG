# corrective_RAG
A Corrective Retrieval-Augmented Generation (RAG) system that uses a step-by-step process to improve accuracy. Built with LangGraph, it brings together document search, relevance scoring, query rewriting, and even web search to deliver more complete and reliable answers.

Features
Document Upload & Processing: Upload PDFs, TXT, or MD files and process them into semantic chunks for retrieval.
Smart Retrieval: Uses ChromaDB with OpenAI Embeddings for semantic document retrieval.
Relevance Grading: Applies an LLM to grade document relevance, ensuring only useful documents are used for answering.
Query Transformation: Optimizes user queries before hitting external sources.
Web Search Fallback: If document coverage is low, it queries Google Custom Search for complementary data.
Final Answer Generation: Uses GPT-4 to generate a comprehensive answer based on validated and enriched context.
Streamlit UI: Interactive interface for document upload, API configuration, and question answering.

How to Run
1. Clone the Repository
git clone https://github.com/kundhana2123/corrective_RAG.git
cd your-repo-directory
2. Install Dependencies
pip install -r requirements.txt
3. Set Your API Keys
OpenAI API Key (for embeddings and GPT-4)
Google API Key and CSE ID (for Google Custom Search)

Run the App
streamlit run main.py
