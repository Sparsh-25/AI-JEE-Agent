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


def translate_to_english(text: str) -> str:

    from langchain_core.messages import HumanMessage, SystemMessage
    try:

        response = llm.invoke([
            SystemMessage(content="""You are a translator. 
    Translate the following text to English.
    Return ONLY the translated text, nothing else.
    No explanations, no notes, just the translation.
    DO NOT ANSWER ANYTHING, JUST TRANSLATE THE GIVEN INTO ENGLISH."""),
            HumanMessage(content=text)
        ])

        return response.content.strip()

    except Exception as e:
        print(f"Translation failed: {e} — using original text")
        return text


# ---- TOOL 1: Search JEE material ----
@tool
def search_jee_material(query: str) -> str:

    """Search the JEE study material for theory, concepts, definitions,
    units, laws, and explanations. Use this for any conceptual or
    theory-based question."""

    from langdetect import detect, LangDetectException

    try:
        # Detect language of the query
        language = detect(query)
    except LangDetectException:
        language = "en"  # default to English if detection fails
    
    # If not English, translate first
    if language != "en":
        print(f"Detected language: {language} — translating to English first")
        english_query = translate_to_english(query)
        print(f"Translated query: {english_query}")
    else:
        english_query = query

    results = vectordb.similarity_search(english_query, k=3)
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
- For calculations use calculator tool with PLAIN NUMBERS ONLY
  - Write 8990000000 instead of 8.99e9
  - Write 8990000000 * 4 / 1 not k*(q1*q2)/r^2
  - Never put variable names or units in the calculator
- IMPORTANT: Always respond in the same language the user asked in.
  If they asked in Hindi, reply in Hindi.
  If they asked in English, reply in English.
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

test_history = []
def test(question):
    global test_history
    answer, test_history = ask_agent(question, test_history)
    print(f"\nQ: {question}")
    print(f"A: {answer}\n")
    print("-" * 40)

test("What is electric field?")                          # English
test("विद्युत क्षेत्र की SI इकाई क्या है?")              # Hindi: What is SI unit of electric field?
test("Coulomb ka niyam samjhao")                        # Hindi: Explain Coulomb's law
test("Calculate force for 2C charges 1m apart") 

