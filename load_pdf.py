from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/Koushali_resume.pdf")
pages = loader.load()
print(f"Total pages loaded: {len(pages)}")
print("\nFirst page content preview:")
print(pages[0].page_content[:700])  # prints first 500 characters