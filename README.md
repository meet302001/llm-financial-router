# Gen-AI Intern Assessment – SEC 10-K QA & Financial Tool Router

This repository is a submission for the Traderware Gen-AI Internship assessment. It demonstrates three advanced capabilities using LangChain and LangGraph:

1. Retrieval-Augmented Generation (RAG) on 10-K SEC filings
2. A LangGraph-based financial tool router
3. Automatic evaluation of QA chain performance with cost monitoring

---

## 📌 Task 1: 10-K Retrieval QA

### ✔ Description
- Downloads the latest **10-K filings** for 10 companies using `sec-edgar-downloader`
- Parses and **chunks** documents using `RecursiveCharacterTextSplitter`
- Stores them in a **FAISS** vectorstore after embedding with `OpenAIEmbeddings`
- Uses `RetrievalQA` from LangChain to answer 10-K-related questions

### 💡 Why These Choices?
- **RecursiveCharacterTextSplitter** was chosen to preserve semantic coherence across paragraphs.
- **FAISS** provides fast similarity search, suitable for scalable retrieval.
- **OpenAI Embeddings** ensure powerful vector representations for unstructured financial data.

---

## 📌 Task 2: LangGraph Financial Tool Router

### ✔ Description
- User queries are routed dynamically to one of:
  - `price_lookup`
  - `news_headlines`
  - `stat_ratios`
- A LangGraph-based **LLM Router** selects the appropriate tool.
- Tool results are composed into a user-friendly response.

### 💡 Why These Choices?
- **LangGraph** allows clean separation of routing, tool logic, and formatting.
- Using an **LLM-based router** supports flexibility in handling natural language queries.
- Modular architecture allows plug-and-play addition of new tools.

---

## 📌 Task 3: Chain Evaluation & Cost Ledger

### ✔ Description
- Evaluates QA accuracy using a test set of known Q-A pairs from Apple's 10-K
- Computes **F1 score** and tracks simulated token usage
- Asserts failure if:
  - Mean F1 score < 0.6
  - Total cost > $0.10

### 💡 Why These Choices?
- **F1 score** is a simple, effective metric for short-answer QA
- **Token budgeting** ensures the solution is scalable and API-cost aware
- **Assertions** enforce quality and efficiency like a CI check

---

## 🧪 Technologies Used
- Python, LangChain v0.2+
- LangGraph
- OpenAI (GPT-3.5-Turbo, Embeddings)
- FAISS
- NumPy, scikit-learn
- `sec-edgar-downloader` for data ingestion

---

## 🚀 Running the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set your OpenAI API Key

You can either:
- Set it via environment variable:
```bash
export OPENAI_API_KEY=sk-xxxxxxxx...
```
- Or, modify the script directly where the key is passed.

### Step 3: Run the pipeline
```bash
python tasks.py
```
This will:
- Download 10-Ks
- Build vectorstore
- Answer financial questions
- Route LangGraph queries
- Evaluate model accuracy and simulated cost

---

## 📂 Folder Structure

```
GenAI_assessment/
├── tasks.py
├── requirements.txt
├── sec-edgar-filings/         # Downloaded 10-Ks
├── README.md
```

---

## 📌 Notes
- All tools are mocked for now (e.g., price and news); can be extended to real APIs.
- Document chunking and embedding are reproducible using a fixed seed.

---

## 📣 Author

Meet Bhanushali — Data Science Graduate Student @ Pace University  
3x AWS Certified | GenAI & LLM Practitioner

