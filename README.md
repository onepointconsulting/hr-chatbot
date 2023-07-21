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
pip install prompt_toolkit
```

### Custom environment

```
# conda activate base
# conda remove -n langchain_chainlit_2 --all
conda create -n langchain_chainlit_2 python=3.11
conda activate langchain_chainlit_2
# pip install --force-reinstall /home/ubuntu/chainlit-0.5.3-py3-none-any.whl
pip install --force-reinstall C:\development\playground\chainlit\src\dist\chainlit-0.5.3-py3-none-any.whl
pip install langchain
pip install faiss-cpu
pip install tiktoken
pip install pdfminer
pip install pypdfium2
pip install black
```

## Configuration

Please make sure that you have a .env file with the following variables:
```
OPENAI_API_KEY=<open ai key>
DOC_LOCATION=<absolute path of the pdf files>
FAISS_STORE=<Location of the FAISS internal files>
HUMOUR=<true|false>
```

## Running

With Chainlit:
```
chainlit run hr_chatbot_chainlit.py --port 8081
```

For Development:
```
chainlit run hr_chatbot_chainlit.py -w --port 8081
```

Command line:
```
python ./hr_chatbot_cli.py
```