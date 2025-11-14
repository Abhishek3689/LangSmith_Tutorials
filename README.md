# 🧠 LangSmith Integration Guide

A clear and production-ready **README.md** for integrating, testing, tracing, and evaluating LLM applications using **LangSmith**.

---

## 🚀 Overview

This project demonstrates how to integrate **LangSmith** into your GenAI or RAG application for:

* **Tracing** requests and responses
* **Observability** of agentic workflows
* **Dataset creation** for model evaluation
* **Automated testing** and quality checks

LangSmith makes it easy to debug and optimize LLM workflows in production.

---

## 📦 Features

* 🔍 **Real-time traces** for every model call
* 🧪 **Evals** using built-in or custom evaluators
* 📚 **Dataset creation** from logs or manual examples
* 🤖 **Agent monitoring** for complex multi-step execution
* 📈 **Performance metrics** and analytics

---

## 🛠️ Installation

```bash
pip install langsmith
```

---

## 🔧 Environment Setup

Create a `.env` file with the following:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_api_key_here
LANGCHAIN_PROJECT=your_project_name
```

Load environment variables:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 📘 Basic Usage

### 1. **Initialize the LangSmith Client**

```python
from langsmith import Client
client = Client()
```

### 2. **Trace a simple LLM call**

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

response = model.invoke("What is LangSmith?")
print(response.content)
```

All calls automatically appear in your LangSmith dashboard.

---

## 📂 Creating Datasets

You can create datasets for evaluation:

```python
dataset = client.create_dataset(
    name="rag_test_questions",
    description="Test dataset for RAG pipeline"
)

client.create_example(
    inputs={"question": "What is the capital of India?"},
    outputs={"answer": "New Delhi"},
    dataset_id=dataset.id,
)
```

---

## 🧪 Running Evaluations

You can run an evaluation on a dataset:

```python
from langsmith.evaluation import evaluate

evaluation_results = evaluate(
    dataset_name="rag_test_questions",
    llm_or_chain_factory=lambda: model,
    evaluators=["qa", "criteria", "embedding_distance"],
)

print(evaluation_results)
```

---

## 🤖 Monitoring Agentic Applications

If you have agents using **LangChain**, all tool calls and reasoning steps are auto-traced.

Example:

```python
from langchain.agents import initialize_agent, load_tools

tools = load_tools(["serpapi"])
agent = initialize_agent(tools, model,
```
