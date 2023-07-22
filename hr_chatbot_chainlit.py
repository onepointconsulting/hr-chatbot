from langchain.chains import RetrievalQAWithSourcesChain
import chainlit as cl

from chain_factory import create_retrieval_chain, load_all_chains, load_embeddinges
from geolocation import extract_ip_address, geolocate
from source_splitter import source_splitter
from chainlit.context import get_emitter

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
        index_html = Path(build_dir) / "index.html"
        if index_html.exists():
            with open(index_html, "r") as f:
                content = f.read()
                hide_css = """
        <style>
            a[href='https://github.com/Chainlit/chainlit'] {
                visibility: hidden;
            }
        </style>
    """
                if "visibility: hidden" not in content:
                    changed_html = re.sub(
                        r"(</head>)", hide_css + "</head>", content, re.MULTILINE
                    )
                    logger.warn(f"Changed HTML {changed_html}")
                    with open(index_html, "w") as f:
                        f.write(changed_html)
    except Exception as e:
        logger.error("Could not process 'built with' styles.")


@cl.langchain_factory(use_async=True)
async def init():
    """
    Loads the vector data store object and the PDF documents. Creates the QA chain.
    Sets up some session variables and removes the Chainlit footer.

    Parameters:
    use_async (bool): Determines whether async is to be used or not.

    Returns:
    RetrievalQAWithSourcesChain: The QA chain
    """

    emitter = get_emitter()
    # Please note this works only with a modified version of Streamlit
    # The repo with this modification are here: https://github.com/gilfernandes/chainlit_hr_extension
    remote_address = extract_ip_address(emitter.session.environ)
    geo_location = geolocate(remote_address)

    country_code = "GB"
    if geo_location.country_code != "Not found":
        country_code = geo_location.country_code
        geo_location_msg = cl.Message(
            content=f"""Geo location: 
- country: {geo_location.country_name}
- country code: {country_code}"""
        )
        await geo_location_msg.send()

    logger.info(f"Geo location: {geo_location}")

    msg = cl.Message(content=f"Processing files. Please wait.")
    await msg.send()
    chain_dict = load_all_chains(country_code)
    qa_data = chain_dict[country_code]

    documents = qa_data.documents

    chain: RetrievalQAWithSourcesChain = qa_data.chain
    metadatas = [d.metadata for d in documents]
    texts = [d.page_content for d in documents]
    cl.user_session.set(KEY_META_DATAS, metadatas)
    cl.user_session.set(KEY_TEXTS, texts)

    msg.content = (
        f"You can now ask questions about Onepoint HR (IP Address: {remote_address}, country code: {country_code})!"
    )
    await msg.send()

    return chain


@cl.langchain_postprocess
async def process_response(res) -> cl.Message:
    """
    Tries to extract the sources and corresponding texts from the sources.

    Parameters:
    res (dict): A dictionary with the answer and sources provided by the LLM via LangChain.

    Returns:
    cl.Message: The message containing the answer and the list of sources with corresponding texts.
    """
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get(KEY_META_DATAS)
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get(KEY_TEXTS)

    found_sources = []
    if sources:
        logger.info(f"sources: {sources}")
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
            answer += f"\n{sources}"

    await cl.Message(content=answer, elements=source_elements).send()


if __name__ == "__main__":
    pass
