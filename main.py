from typing import Dict, TypedDict
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import END, StateGraph
import tempfile
import pprint
import re
import json
import chromadb
import requests
import os
import streamlit as st

client = chromadb.Client()

#initializing session state
def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.openai_api_key = ""
        st.session_state.google_api_key = ""
        st.session_state.google_cse_id = ""
        st.session_state.ready_to_proceed = False
        st.session_state.docs_uploaded = False
        st.session_state.retriever = None
        st.session_state.documents = []

initialize_session_state()

#sidebar for entering API and then validating 
def setup_sidebar():
    with st.sidebar:
        st.subheader("API Configuration")

        st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key)
        st.session_state.google_api_key = st.text_input("Google API Key", type="password", value=st.session_state.google_api_key)
        st.session_state.google_cse_id = st.text_input("Google CSE ID", type="password", value=st.session_state.google_cse_id)

        if st.button("Proceed"):
            if all([
                st.session_state.openai_api_key.strip(),
                st.session_state.google_api_key.strip(),
                st.session_state.google_cse_id.strip()
            ]):
                try:
                    # testing openai API
                    test_llm = ChatOpenAI(
                        temperature=0,
                        model="gpt-4",
                        openai_api_key=st.session_state.openai_api_key
                    )
                    test_llm.invoke("Hello")

                    #now testing google api
                    test_response = requests.get(
                        "https://www.googleapis.com/customsearch/v1",
                        params={
                            "key": st.session_state.google_api_key,
                            "cx": st.session_state.google_cse_id,
                            "q": "test"
                        },
                        timeout=5
                    )
                    test_response.raise_for_status()

                    st.session_state.ready_to_proceed = True
                    st.success("✅ All API keys are valid. You may proceed.")
                except Exception as e:
                    st.session_state.ready_to_proceed = False
                    st.error(f"API key validation failed: {e}")
            else:
                st.error("Please enter all three API keys.")

setup_sidebar()

if not st.session_state.ready_to_proceed:
    st.info("Please enter and validate your API keys in the sidebar to proceed.")
    st.stop()

# 
st.subheader("Corrective - RAG")
uploaded_file = st.file_uploader("Upload Document",type=['pdf', 'txt', 'md'])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    def load_documents(file_path: str) -> list:
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext in ['.txt', '.md']:
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            return loader.load()
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            return []

    docs = load_documents(tmp_path)
    os.unlink(tmp_path)

    if docs:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=st.session_state.openai_api_key)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
        all_splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(all_splits, embeddings, client=client)
        retriever = vectorstore.as_retriever()
        st.session_state.retriever = retriever
        st.session_state.documents = docs
        st.session_state.docs_uploaded = True
        st.success("Document loaded and processed")
else:
    if not st.session_state.docs_uploaded:
        st.info("Please upload a document to proceed.")

if st.session_state.docs_uploaded:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=st.session_state.openai_api_key)
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=st.session_state.openai_api_key)

    class GraphState(TypedDict):
        keys: Dict[str, any]


    def retrieve(state):
        state_dict = state["keys"]
        question = state_dict["question"]
        retriever = st.session_state.get("retriever", None)
        if retriever is None:
            return {"keys": {"documents": [], "question": question}}
        documents = retriever.get_relevant_documents(question)
        return {"keys": {"documents": documents, "question": question}}

    def grade_documents(state):
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        prompt = PromptTemplate(template="""You are grading the relevance of a retrieved document to a user question.
Return ONLY a JSON object with a "score" field representing the confidence that the document is relevant.
The "score" should be a number between 0 (not relevant) and 1 (fully relevant).
Do not include any other text or explanation.

Document: {context}
Question: {question}

Rules:
- Consider keyword matches and semantic similarity.
- Be lenient but precise in scoring.
- Return exactly like this example: {{"score": 0.85}}""", input_variables=["context", "question"])

        chain = prompt | llm | StrOutputParser()
        filtered_docs = []
        search = "No"
        threshold = 0.7

        for d in documents:
            try:
                response = chain.invoke({"question": question, "context": d.page_content})
                try:
                    score_obj = json.loads(response)
                except json.JSONDecodeError:
                    continue

                score = float(score_obj.get("score", 0))
                if score >= threshold:
                    filtered_docs.append(d)
                else:
                    search = "Yes"
            except Exception:
                filtered_docs.append(d)
                continue

        return {"keys": {"documents": filtered_docs, "question": question, "run_web_search": search}}

    def transform_query(state):
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        prompt = PromptTemplate(template="""Generate a search-optimized version of this question by analyzing its core semantic meaning and intent.
-------
{question}
-------
Return only the improved question with no additional text:""", input_variables=["question"])

        chain = prompt | llm | StrOutputParser()
        better_question = chain.invoke({"question": question})
        return {"keys": {"documents": documents, "question": better_question}}

    def web_search(state):
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": st.session_state["google_api_key"],
            "cx": st.session_state["google_cse_id"],
            "q": question,
            "num": 5
        }

        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            results = response.json().get("items", [])
        except Exception as e:
            st.error(f"Error calling Google Custom Search API: {e}")
            results = []

        web_results = []
        for result in results:
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No content")
            link = result.get("link", "No URL")
            content = f"Title: {title}\nURL: {link}\nSnippet: {snippet}"
            web_results.append(content)

        if web_results:
            web_document = Document(
                page_content="\n\n".join(web_results),
                metadata={
                    "source": "google_search",
                    "query": question,
                    "result_count": len(web_results)
                }
            )
            documents.append(web_document)

        return {"keys": {"documents": documents, "question": question}}

    def generate(state):
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        prompt = PromptTemplate(template="""Based on the following context, please answer the question.
Context: {context}
Question: {question}
Answer:""", input_variables=["context", "question"])

        context = "\n\n".join(doc.page_content for doc in documents)
        rag_chain = (
            {"context": lambda x: context, "question": lambda x: question}
            | prompt
            | llm
            | StrOutputParser()
        )
        generation = rag_chain.invoke({})
        return {"keys": {"documents": documents, "question": question, "generation": generation}}

    def decide_to_generate(state):
        state_dict = state["keys"]
        search = state_dict.get("run_web_search", "No")
        if search == "Yes":
            return "transform_query"
        else:
            return "generate"

    def format_document(doc: Document) -> str:
        return f"Source: {doc.metadata.get('source', 'Unknown')}\nTitle: {doc.metadata.get('title', 'No title')}\nContent: {doc.page_content[:200]}..."

    def format_state(state: dict) -> str:
        formatted = {}
        for key, value in state.items():
            if key == "documents":
                formatted[key] = [format_document(doc) for doc in value]
            else:
                formatted[key] = value
        return formatted

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {"transform_query": "transform_query", "generate": "generate"})
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    app = workflow.compile()

    st.subheader("Ask a question:")
    user_question = st.text_input("Enter your question here:")

    if user_question:
        inputs = {"keys": {"question": user_question}}
        final_generation = "No final answer generated."
        for output in app.stream(inputs):
            for value in output.values():
                final_generation = value["keys"].get("generation", final_generation)
        st.subheader("Answer:")
        st.write(final_generation)
    else:
        st.stop()