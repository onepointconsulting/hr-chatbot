# Onepoint HR Chatbot

This is a simple HR chatbot based on Chainlit with memory support.

## Installation

```
conda create -n langchain_chainlit python=3.11
conda activate langchain_chainlit
pip install langchain
pip install python-dotenv
pip install openai
pip install faiss-cpu
pip install tiktoken
pip install chainlit
pip install pdfminer
pip install pypdfium2
```

## Running

With Chainlit:
```
chainlit run hr_chatbot_chainlit.py
```

For Development:
```
chainlit run hr_chatbot_chainlit.py -w --port 8081
```