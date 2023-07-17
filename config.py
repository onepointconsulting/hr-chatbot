from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()

class Config():
    chunk_size = 6000
    chunk_overlap = 100
    chunk_separator = "\n\n"
    faiss_persist_directory = Path(os.environ['FAISS_STORE'])
    if not faiss_persist_directory.exists():
        faiss_persist_directory.mkdir()
    embeddings = OpenAIEmbeddings(chunk_size=100)
    model = 'gpt-3.5-turbo-16k'
    # model = 'gpt-4'
    llm = ChatOpenAI(model=model, temperature=0)
    search_results = 5

    def __repr__(self) -> str:
        return (
f"""# Configuration
chunk_size: {self.chunk_size}
chunk_overlap: {self.chunk_overlap}
chunk_separator: {self.chunk_separator}
faiss_persist_directory: {self.faiss_persist_directory}

embeddings: {self.embeddings}

llm: {self.llm}
""")

cfg = Config()

if __name__ == "__main__":
    print(cfg)