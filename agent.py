import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Load vector database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory='./chroma_db'
)

llm = ChatGroq(model="llama-3.1-8b-instant")

# ---- TOOL 1: Search JEE material ----
@tool
def search_jee_material(query: str) -> str:
    """Search the JEE study material for theory, concepts, definitions,
    units, laws, and explanations. Use this for any conceptual or
    theory-based question."""
    results = vectordb.similarity_search(query, k=3)
    if not results:
        return "No relevant content found in the study material."
    return "\n\n".join([doc.page_content for doc in results])

# ---- TOOL 2: Calculator ----
@tool
def calculator(expression: str) -> str:
    """Use this for any mathematical calculation or numerical problem.
    Input must be a valid math expression like '9.8 * 1200' or '0.5 * 10 * 5**2'."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

# ---- BUILD AGENT ----
tools = [search_jee_material, calculator]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="""You are a helpful JEE tutor. 

Rules:
- ALWAYS use search_jee_material first before answering any theory question
- If the material doesn't contain the answer, say 'This topic isn't in my notes yet' and give a brief general answer
- For any calculation use calculator tool, never calculate in your head
- Keep answers concise and JEE-focused"""
)


def ask_agent(question, history):
    print(f"\n{'='*50}")
    print(f"You: {question}")
    print('='*50)

    # Add user's new question to history
    history.append({"role": "user", "content": question})

    # Send full history to agent every time
    result = agent.invoke({"messages": history})

    # Extract final answer
    final_answer = result["messages"][-1].content

    # Save agent's reply to history so next question has context
    history.append({"role": "assistant", "content": final_answer})

    print(f"\nAgent: {final_answer}")
    return final_answer, history


