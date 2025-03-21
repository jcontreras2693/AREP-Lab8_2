# Taller 8.2 | AREP

# README: Building a Retrieval Augmented Generation (RAG) Application

## Overview

This repository demonstrates how to build a **Retrieval Augmented Generation (RAG)** application using **LangChain**. The application retrieves relevant information from a document store and generates answers to user queries using a language model. This project is divided into two parts:

1. **Part 1**: Focuses on setting up the RAG pipeline, including document loading, splitting, indexing, and retrieval.
2. **Part 2**: Expands the application by adding stateful chat history management and agent-based query handling.

By following this guide, you will learn:

- How to load, split, and index documents for retrieval.
- How to create a RAG pipeline using LangChain.
- How to manage stateful chat history and handle complex queries with agents.

---

## Project Architecture and Components

The application consists of the following components:

1. **Document Loading**: Loads documents from a web source (e.g., a blog post).
2. **Text Splitting**: Splits documents into smaller chunks for efficient retrieval.
3. **Vector Store**: Stores document embeddings for similarity search.
4. **Retrieval**: Retrieves relevant document chunks based on user queries.
5. **Generation**: Uses a language model to generate answers based on retrieved content.
6. **State Management**: Manages chat history and context for multi-turn conversations.
7. **Agent**: Handles complex queries by breaking them into smaller tasks.

---

## Installation

### Prerequisites

- Python 3.7 or higher.
- OpenAI API key.
- Pinecone API key (for vector storage).

### Step-by-Step Instructions

1. **Install Required Libraries**:
Install the necessary libraries using `pip`:
    
    ```bash
    pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph beautifulsoup4 langchain-openai langchain-pinecone pinecone-notebooks
    
    ```
    
2. **Set Up API Keys**:
Set your OpenAI and Pinecone API keys as environment variables.
    
    ### OpenAI API Key:
    
    ```python
    import getpass
    import os
    
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    
    ```
    
    ### Pinecone API Key:
    
    ```python
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
    
    ```
    
3. **Initialize Pinecone Index**:
Create a Pinecone index for storing document embeddings.
    
    ```python
    from pinecone import Pinecone, ServerlessSpec
    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = "langchain-test-index"  # Change if desired
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    
    index = pc.Index(index_name)
    
    ```
    
4. **Initialize Embeddings**:
Use OpenAI embeddings for document encoding.
    
    ```python
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    ```
    
5. **Set Up Vector Store**:
Use Pinecone as the vector store.
    
    ```python
    from langchain_pinecone import PineconeVectorStore
    
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    ```
    

---

## Running the Code

### Part 1: Basic RAG Pipeline

1. **Load and Split Documents**:
Load a blog post and split it into smaller chunks.
    
    ```python
    import bs4
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    loader = WebBaseLoader(
        web_paths=("<https://lilianweng.github.io/posts/2023-06-23-agent/>",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    )
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    
    ```
    
2. **Index Documents**:
Add document chunks to the vector store.
    
    ```python
    document_ids = vector_store.add_documents(documents=all_splits)
    
    ```
    
3. **Define RAG Prompt**:
Use a predefined prompt for question-answering.
    
    ```python
    from langchain import hub
    
    prompt = hub.pull("rlm/rag-prompt")
    
    ```
    
4. **Retrieve and Generate**:
Retrieve relevant documents and generate answers.
    
    ```python
    from langchain_core.documents import Document
    from typing_extensions import List, TypedDict
    
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str
    
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}
    
    def generate(state: State):
        docs_content = "\\n\\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    from langgraph.graph import START, StateGraph
    
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    result = graph.invoke({"question": "What is Task Decomposition?"})
    print(result["answer"])
    
    ```
    

---

### Part 2: Stateful Chat History and Agents

1. **Stateful Management**:
Use a memory checkpoint to manage chat history.
    
    ```python
    from langgraph.checkpoint.memory import MemorySaver
    
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    
    ```
    
2. **Agent-Based Query Handling**:
Use an agent to handle complex queries.
    
    ```python
    from langgraph.prebuilt import create_react_agent
    
    agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
    
    ```
    
3. **Run the Agent**:
Stream responses from the agent.
    
    ```python
    config = {"configurable": {"thread_id": "abc123"}}
    
    for event in agent_executor.stream(
        {"messages": [{"role": "user", "content": "What is Task Decomposition?"}]},
        stream_mode="values",
        config=config,
    ):
        event["messages"][-1].pretty_print()
    
    ```
    

---

## Example Output

### Input:

```python
result = graph.invoke({"question": "What is Task Decomposition?"})
print(result["answer"])

```

### Output:

```
Task Decomposition is the process of breaking down a complicated task into smaller, manageable steps. It involves techniques like Chain of Thought (CoT), where the model is prompted to "think step by step," and can include various methods such as simple prompting, task-specific instructions, or human inputs.

```

---

## Screenshots

![image](https://github.com/user-attachments/assets/f2a8963a-1a3f-4778-94d2-8e898072a335)

![image](https://github.com/user-attachments/assets/29fe899f-6d4a-4d15-83fc-09e96b1290e0)

---

## Conclusion

This project demonstrates how to build a RAG application using LangChain. You’ve learned how to:

- Load, split, and index documents.
- Retrieve relevant information and generate answers.
- Manage stateful chat history and handle complex queries with agents.

This is just the beginning! LangChain offers many more features for building advanced AI applications. For further learning, check out the [LangChain Documentation](https://langchain.com/docs).

---

## Acknowledgement

* [LangChain Tutorial](https://python.langchain.com/docs/tutorials/rag/#splitting-documents)
* [LangChain Pinecone Tutorial](https://python.langchain.com/docs/integrations/vectorstores/pinecone/)

## Authors

* **Juan David Contreras Becerra** - *Taller 8.2 | AREP* - [AREP-Lab8.2](https://github.com/jcontreras2693/AREP-Lab8_2.git)
