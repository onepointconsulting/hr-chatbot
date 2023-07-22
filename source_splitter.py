import re
from log_init import logger


def source_splitter(sources: str):
    logger.info(f"There are sources: {sources}")
    raw_sources, file_sources = [], []
    split_char = "," if "," in sources else "-"
    for source in sources.split(split_char):
        source = source.strip()
        raw_sources.append(source)
        file_sources.append(re.sub(r"(.+\.pdf).*", r"\1", source))
    return raw_sources, file_sources


if __name__ == "__main__":
    sources = "04.16 Code of Conduct (1).pdf page 1, 04.13 HR Policies & Procedures V10.docx .pdf page 1"
    raw_sources, file_sources = source_splitter(sources)
    logger.info(f"raw sources: {raw_sources}")
    logger.info(f"file sources: {file_sources}")
    print()
    sources = """Family Friendly Rights & Policies V1.2.pdf page 16, page 22"""
    raw_sources, file_sources = source_splitter(sources)
    logger.info(f"raw sources: {raw_sources}")
    logger.info(f"file sources: {file_sources}")
