from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # new, correct


load_dotenv()


# Load PDFs
all_chunks = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
      )

for filename in os.listdir("data"):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(f"data/{filename}")
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"Loaded {filename}: {len(chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")


print('\nLoading embedding model.....')

embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

vectordb = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Done. {vectordb._collection.count()} chunks stored in database.")
print("Do not run this file again unless you add new PDFs.")





