from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import shutil

# ===== CHANGE ONLY THIS PART =====

PDF_PATH = "book3.pdf"
DB_PATH = "oxford_hb_db"

COLLECTION_NAME = "book3_1500"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = int(0.2*CHUNK_SIZE)
START_PAGE = 15

BOOK_TITLE = "Oxford Handbook of Clinical Medicine"

# ================================

def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    return text.strip()

loader = PyMuPDFLoader(PDF_PATH)
docs = loader.load()
docs = docs[START_PAGE:]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

split_docs = splitter.split_documents(docs)

clean_docs = []

for doc in split_docs:
    text = clean_text(doc.page_content)

    if 50 <= len(text) <= 5000:
        doc.page_content = text
        clean_docs.append(doc)

for doc in clean_docs:
    doc.metadata["book"] = BOOK_TITLE

print("Original chunks:", len(split_docs))
print("Clean chunks:", len(clean_docs))

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DB_PATH
)

batch_size = 20

for i in range(0, len(clean_docs), batch_size):
    batch = clean_docs[i:i + batch_size]

    try:
        print(f"Adding batch {i} to {i + len(batch)}")
        vectorstore.add_documents(batch)
        print("Batch added successfully")

    except Exception as e:
        print("\nFAILED BATCH")
        print("Batch range:", i, "to", i + len(batch))
        print("Error:", e)

        for j, doc in enumerate(batch):
            print(f"\n--- BAD CHECK {i + j} ---")
            print(doc.page_content[:1000])

        break

print("Done")
