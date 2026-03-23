from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
 
llm = ChatGroq(model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_messages([

    ('system', 'You are a JEE tutor.Answer clearly and concisely'),
    ('human', '{question}')
])

chain = prompt | llm

response = chain.invoke({'question': 'Explain 2nd law of thermodynamics'})

print(response.content)