from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory.utils import get_prompt_input_key
from config import cfg
from typing import Any, Dict, Tuple, List, TypeVar


import os
from pathlib import Path

from generate_embeddings import load_pdfs, generate_embeddings

from log_init import logger

VST = TypeVar("VST", bound="VectorStore")


class KeySourceMemory(ConversationSummaryBufferMemory):

    def _get_input_output(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            output_key = 'answer'
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]


def load_embeddinges() -> Tuple[VST, List[Document]]:
    embedding_dir = cfg.faiss_persist_directory
    logger.info(f"Checking: {embedding_dir}")
    doc_location: str = os.environ["DOC_LOCATION"]
    documents = load_pdfs(Path(doc_location))
    assert len(documents) > 0
    if embedding_dir.exists() and len(list(embedding_dir.glob("*"))) > 0:
        logger.info(f"reading from existing directory: {embedding_dir}")
        docsearch = FAISS.load_local(embedding_dir, cfg.embeddings)
        return docsearch, documents
    return generate_embeddings(documents, doc_location), documents


def create_retrieval_chain(docsearch: VST) -> RetrievalQAWithSourcesChain:
    # Create a chain that uses the Chroma vector store

    memory = KeySourceMemory(llm=cfg.llm, input_key='question', output_key='answer')
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        cfg.llm, 
        retriever=docsearch.as_retriever(),
        chain_type="stuff", 
        memory=memory
    )

    return qa_chain


if __name__ == "__main__":
    docsearch, documents = load_embeddinges()
    chain: RetrievalQAWithSourcesChain = create_retrieval_chain(docsearch)
    logger.info(chain)


    