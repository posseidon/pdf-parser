from file.loader import ParallelFileLoader, clean_text
from sml.model import SmallLanguageModel

def parse_pdf_file(filename: str) -> str:
    """Parse the content of a file and return cleaned text."""
    loader = ParallelFileLoader(filename)
    content = loader.extract_text()
    return clean_text(content)

def main():
    llm = SmallLanguageModel()

    file_content2 = parse_pdf_file("tests/resources/Hungary_short_history.pdf")
    answer3 = llm.answer_question("Mikor volt a honfoglalás?", file_content2)
    answer4 = llm.answer_question("Kik alkották az első magyar kormányt?", file_content2)
    print(answer3)
    print(answer4)
    

if __name__ == "__main__":
    main()
