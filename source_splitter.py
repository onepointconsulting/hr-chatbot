import re
from log_init import logger


def source_splitter(sources: str):
    logger.info(f"There are sources: {sources}")
    raw_sources, file_sources = [], []
    split_char = "," if "," in sources else "-"
    for source in sources.split(split_char):
        source = source.strip()
        raw_sources.append(source)
        file_sources.append(re.sub(r".+[\/](.+)\spage.+", r"\1", source))
    return raw_sources, file_sources


if __name__ == "__main__":
    sources = "C:\\development\\playground\\langchain\\hr_chatbot\\data\\04.16 Code of Conduct (1).pdf, C:\\development\\playground\\langchain\\hr_chatbot\\data\\04.13 HR Policies & Procedures V10.docx .pdf"
    raw_sources, file_sources = source_splitter(sources)
    logger.info(f"raw sources: {raw_sources}")
    logger.info(f"file sources: {file_sources}")
    sources = """C:\\development\\playground\\langchain\\hr_chatbot\\data\\11.A.8.2 Acceptable Usage Policy.pdf page 10, 11.A.6.1 Mobile Devices_ BYOD and Remote Working Policy.pdf page 5, 7, 11, 04.13 HR Policies & Procedures V10.docx .pdf page 44"""
    raw_sources, file_sources = source_splitter(sources)
    logger.info(f"raw sources: {raw_sources}")
    logger.info(f"file sources: {file_sources}")

