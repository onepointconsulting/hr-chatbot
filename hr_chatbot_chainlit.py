from langchain.chains import RetrievalQAWithSourcesChain
import chainlit as cl

from chain_factory import create_retrieval_chain, load_embeddinges
from source_splitter import source_splitter

from log_init import logger

from pathlib import Path
import re
import os

KEY_META_DATAS = "metadatas"
KEY_TEXTS = "texts"

def remove_footer():
    try:
        build_dir = cl.server.build_dir
        logger.warn(f"Build directory: {build_dir}")
        index_html = Path(build_dir)/'index.html'
        if index_html.exists():
            with open(index_html, 'r') as f:
                content = f.read()
                hide_css = """
        <style>
            a[href='https://github.com/Chainlit/chainlit'] {
                visibility: hidden;
            }
        </style>
    """
                if "visibility: hidden" not in content:
                    changed_html = re.sub(r"(</head>)", hide_css + "</head>", content, re.MULTILINE)
                    logger.warn(f"Changed HTML {changed_html}")
                    with open(index_html, 'w') as f:
                        f.write(changed_html)
    except Exception as e:
        logger.error("Could not process 'built with' styles.")



@cl.langchain_factory(use_async=True)
async def init():

    msg = cl.Message(content=f"Processing files. Please wait.")
    await msg.send()
    docsearch, documents = load_embeddinges()
    
    humour = os.getenv("HUMOUR") == "true"
    
    chain: RetrievalQAWithSourcesChain = create_retrieval_chain(docsearch, humour=humour)
    metadatas = [d.metadata for d in documents]
    texts = [d.page_content for d in documents]
    cl.user_session.set(KEY_META_DATAS, metadatas)
    cl.user_session.set(KEY_TEXTS, texts)
    remove_footer()
    await msg.update(content=f"You can now ask questions about Onepoint HR!")#
    return chain


@cl.langchain_postprocess
async def process_response(res):
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get(KEY_META_DATAS)
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get(KEY_TEXTS)

    found_sources = []
    if sources:
        raw_sources, file_sources = source_splitter(sources)
        for i, source in enumerate(raw_sources):
            try:
                index = all_sources.index(source)
                text = texts[index]
                source_name = file_sources[i]
                found_sources.append(source_name)
                # Create the text element referenced in the message
                logger.info(f"Found text in {source_name}")
                source_elements.append(cl.Text(content=text, name=source_name))
            except ValueError:
                continue
        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()

if __name__ == "__main__":
    pass


    