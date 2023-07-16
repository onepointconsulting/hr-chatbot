from langchain.chains import RetrievalQAWithSourcesChain
from prompt_toolkit import HTML, prompt, PromptSession
from prompt_toolkit.history import FileHistory

from chain_factory import create_retrieval_chain, load_embeddinges

from log_init import logger

if __name__ == "__main__":

    session = PromptSession(history=FileHistory(".agent-history-file"))
    docsearch, documents = load_embeddinges()
    chain: RetrievalQAWithSourcesChain = create_retrieval_chain(docsearch, humor=True)

    while True:
        question = session.prompt(
            HTML("<b>Type <u>Your question</u></b>  ('q' to exit, 's' to save to html file): ")
        )
        if question.lower() == 'q':
            break
        response = chain({'question': question})
        logger.info(response['answer'])
        logger.info(response['sources'])
