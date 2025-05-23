# Corrective_RAG
A Corrective Retrieval-Augmented Generation (RAG) system that uses a step-by-step process to improve accuracy. Built with LangGraph, it brings together document search, relevance scoring, query rewriting, and even web search to deliver more complete and reliable answers.

## Features
#### 1. Document Upload & Processing: <br> 
Upload PDFs, TXT, or MD files and process them into semantic chunks for retrieval.<br>
#### 2. Smart Retrieval: <br>
Uses ChromaDB with OpenAI Embeddings for semantic document retrieval.<br>
#### 3. Relevance Grading: <br>
Applies an LLM to grade document relevance, ensuring only useful documents are used for answering.<br>
#### 4. Query Transformation: <br>
Optimizes user queries before hitting external sources.<br>
#### 5. Web Search Fallback: <br>
If document coverage is low, it queries Google Custom Search for complementary data.<br>
#### 6. Final Answer Generation: <br>
Uses GPT-4 to generate a comprehensive answer based on validated and enriched context.<br>
#### 7. Streamlit UI: <br>
Interactive interface for document upload, API configuration, and question answering.<br>

## How to Run
#### 1. Clone the Repository
git clone https://github.com/kundhana2123/corrective_RAG.git<br>
cd corrective_RAG
#### 2. Install Dependencies <br>
pip install -r requirements.txt
#### 3. Set Your API Keys
OpenAI API Key (for embeddings and GPT-4)<br>
Google API Key and CSE ID (for Google Custom Search)

## Run the app
streamlit run main.py
