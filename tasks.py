import os
import re
import random
import sklearn
import numpy as np
import torch
import json
from sec_edgar_downloader import Downloader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import TypedDict

def download_10k_filings(companies, company_name, email):
    dl = Downloader(company_name, email)
    for ticker in companies:
        dl.get("10-K", ticker)

def load_and_chunk_documents(companies, base_path=None):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for ticker in companies:
        file_path = fr"C:\Users\meetb\Desktop\GenAI_assessment\sec-edgar-filings\{ticker}\10-K"
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if file.endswith(".txt"):
                    loader = TextLoader(os.path.join(root, file))
                    raw_docs = loader.load()
                    docs = text_splitter.split_documents(raw_docs)
                    for doc in docs:
                        doc.metadata["ticker"] = ticker
                    all_docs.extend(docs)
    return all_docs

def create_vectorstore(docs, openai_api_key):
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = None
    batch_size = 50

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        if not vectorstore:
            vectorstore = FAISS.from_documents(batch, embedding)
        else:
            vectorstore.add_documents(batch)

    return vectorstore

def create_qa_chain(vectorstore, model_name="gpt-3.5-turbo"):
    llm = ChatOpenAI(model_name=model_name)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(k=3))

def run_demo_questions(qa_chain):
    questions = [
        "What does Apple list as its three primary sources of revenue?",
        "Summarize the biggest risk Apple cites about supply chain concentration."
    ]
    for q in questions:
        print("Q:", q)
        print("A:", qa_chain.run(q))

class ToolInput(TypedDict):
    ticker: str

def price_lookup(ticker: str) -> str:
    return f"Mock price for {ticker} is $123.45"

def news_headlines(ticker: str, n: int = 3) -> str:
    return f"Top {n} news headlines for {ticker}: [...mock headlines...]"

def stat_ratios(ticker: str) -> str:
    return f"{ticker} - P/E: 24.5, P/S: 9.8, ROE: 15%"

router_prompt = PromptTemplate.from_template("""
Given a user query, choose which of the following tools to use:
- price_lookup
- news_headlines
- stat_ratios

Return a JSON object in the format:
{{"tool": <tool_name>, "args": {{"ticker": "<ticker>"}}}}

Query: {query}
""")

def build_langgraph_router(api_key):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
    router_chain = LLMChain(llm=llm, prompt=router_prompt)

    def router_node(state):
        query = state["messages"][-1].content
        result = router_chain.run({"query": query})
        parsed = json.loads(result)
        return {"tool": parsed["tool"], "args": parsed["args"]}

    def composer_node(state):
        tool_result = state["tool_result"]
        return {"messages": state["messages"] + [AIMessage(content=tool_result)]}

    def tool_node(state):
        tool = state["tool"]
        args = state["args"]
        if tool == "price_lookup":
            return {"tool_result": price_lookup(**args)}
        elif tool == "news_headlines":
            return {"tool_result": news_headlines(**args)}
        elif tool == "stat_ratios":
            return {"tool_result": stat_ratios(**args)}
        else:
            return {"tool_result": "Invalid tool specified."}

    workflow = StateGraph()
    workflow.add_node("router", router_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("composer", composer_node)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "tool")
    workflow.add_edge("tool", "composer")
    workflow.add_edge("composer", END)

    app = workflow.compile()
    return app

def run_langgraph_demo(api_key):
    graph_app = build_langgraph_router(api_key)
    user_query = "What is the P/E ratio of AAPL?"
    result = graph_app.invoke({"messages": [HumanMessage(content=user_query)]})
    print("LangGraph Output:", result["messages"][-1].content)

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9 ]', '', text.lower())

def compute_f1(pred, true):
    pred_tokens = clean_text(pred).split()
    true_tokens = clean_text(true).split()
    if not pred_tokens or not true_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    true_set = set(true_tokens)
    common = pred_set & true_set
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    return 2 * (precision * recall) / (precision + recall)

def evaluate_chain(qa_chain):
    qa_tests = [
        {"question": "What are Apple’s three primary revenue sources?", "answer": "iPhone, Mac, Services"},
        {"question": "Summarize Apple's risk on supply chain concentration", "answer": "Dependence on few suppliers for key components."}
    ]

    total_tokens = 0
    f1_scores = []

    for test in qa_tests:
        response = qa_chain.run(test["question"])
        f1 = compute_f1(response, test["answer"])
        f1_scores.append(f1)
        tokens = 500  # Simulated token usage
        total_tokens += tokens
        print(f"Q: {test['question']} | F1: {f1:.2f} | Tokens: {tokens}")

    mean_f1 = np.mean(f1_scores)
    total_cost = total_tokens * 0.000002  # $0.002 per 1K tokens

    if mean_f1 < 0.6 or total_cost > 0.10:
        raise AssertionError("Model failed minimum score/cost thresholds.")

    print(f"\nMean F1: {mean_f1:.2f}, Total Cost: ${total_cost:.4f}")

def main():
    companies = ["AAPL"]#, "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "DIS"]
    company_name = "YourCompanyName"
    email = "nazstud123@gmail.com"
    openai_api_key = "YOUR_API_KEY"

    download_10k_filings(companies, company_name, email)
    docs = load_and_chunk_documents(companies)
    vectorstore = create_vectorstore(docs, openai_api_key)
    qa_chain = create_qa_chain(vectorstore)
    run_demo_questions(qa_chain)
    evaluate_chain(qa_chain)
    run_langgraph_demo(openai_api_key)

if __name__ == "__main__":
    main()