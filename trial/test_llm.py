from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

response = llm.invoke([HumanMessage(content="What is Newton's third law?")])

print(response.content)