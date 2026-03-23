from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory='./chroma_db'
)

llm = ChatGroq(
    model= "llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a JEE tutor. Answer using the only context below if the answer isn't in context, the answer in the context should be well written in context, say 'I do not have any information regarding this question'
    
    Context:
    {context}"""),

    ("human", "{question}")

])


def ask(question):
    chunks = vectordb.similarity_search(question, k=3)
    context = "\n\n".join([chunk.page_content for chunk in chunks])
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content

