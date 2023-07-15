import re
from log_init import logger


def source_splitter(sources: str):
    logger.info(f"There are sources: {sources}")
    raw_sources, file_sources = [], []
    for source in sources.split(","):
        source = source.strip()
        raw_sources.append(source)
        file_sources.append(re.sub(r".+[\\/](.+)", r"\1", source))
    return raw_sources, file_sources


if __name__ == "__main__":
    sources = "C:\\development\\playground\\langchain\\hr_chatbot\\data\\04.16 Code of Conduct (1).pdf, C:\\development\\playground\\langchain\\hr_chatbot\\data\\04.13 HR Policies & Procedures V10.docx .pdf"
    raw_sources, file_sources = source_splitter(sources)
    logger.info(f"raw sources: {raw_sources}")
    logger.info(f"file sources: {file_sources}")
