import warnings
warnings.filterwarnings("ignore")

import re
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory='./chroma_db'
)

llm = ChatGroq(model="openai/gpt-oss-120b")

def detect_language(text: str) -> str:
    devanagari = re.compile(r'[\u0900-\u097F]')
    if devanagari.search(text):
        return "hi"

    hindi_words = [
        "kya", "hai", "mein", "aur", "nahi",
        "kaise", "kyun", "niyam", "sutra", "urja", "bal", "shakti",
        "doosra", "pehla", "batao", "samjhao",
        "vidyut", "kshetra", "aavesh", "dhaara", "teesra"
    ]
    words = text.lower().split()
    if sum(1 for w in words if w in hindi_words) >= 2:
        return "hi"
    return "en"

def translate_to_english(text: str) -> str:
    try:
        response = llm.invoke([
            SystemMessage(content="""You are a translator. Your ONLY job is to translate text to English.
DO NOT answer the question.
DO NOT add explanation.
Just translate the words. Nothing else.

Example:
Input: विद्युत क्षेत्र की SI इकाई क्या है?
Output: What is the SI unit of electric field?"""),
            HumanMessage(content=f"Translate this to English: {text}")
        ])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text

def search_jee_material(query: str) -> str:
    language = detect_language(query)

    if language != "en":
        logger.info(f"Detected Hindi — translating query")
        english_query = translate_to_english(query)
        logger.info(f"Translated: {english_query}")
    else:
        english_query = query

    results = vectordb.similarity_search(english_query, k=3)

    if not results:
        return "No relevant content found in the study material."

    return "\n\n".join([doc.page_content for doc in results])

def calculator(expression: str) -> str:
    try:
        cleaned = expression.strip()
        cleaned = cleaned.replace('^', '**')
        cleaned = cleaned.replace('×', '*')
        cleaned = cleaned.replace('÷', '/')
        cleaned = re.sub(r'[a-df-wyzA-DF-WYZ]', '', cleaned)
        result = eval(cleaned)
        return f"Result: {round(float(result), 6)}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

def route_question(question: str, history: list) -> str:
    system = """You are a JEE tutor assistant. For the given question decide what to do:

1. If it needs theory/concepts/definitions → call SEARCH
2. If it needs calculation → call CALCULATE
3. If it needs both → call BOTH

Respond in this exact format only:
ACTION: SEARCH or CALCULATE or BOTH
SEARCH_QUERY: (the query to search, in English — only if ACTION is SEARCH or BOTH)
MATH_EXPRESSION: (only numbers and operators, no variables — only if ACTION is CALCULATE or BOTH)

Examples:
Q: What is kinetic energy?
ACTION: SEARCH
SEARCH_QUERY: kinetic energy definition formula

Q: Calculate 0.5 * 10 * 25
ACTION: CALCULATE
MATH_EXPRESSION: 0.5 * 10 * 25

Q: What is KE and calculate it for 10kg at 5m/s
ACTION: BOTH
SEARCH_QUERY: kinetic energy definition formula
MATH_EXPRESSION: 0.5 * 10 * 5**2"""

    messages = [SystemMessage(content=system)]

    for msg in history[-4:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(SystemMessage(content=f"Previous answer: {msg['content'][:200]}"))

    messages.append(HumanMessage(content=f"Question: {question}"))

    response = llm.invoke(messages)
    return response.content

def parse_routing(routing_text: str) -> dict:
    result = {"action": "SEARCH", "search_query": None, "math_expression": None}

    for line in routing_text.strip().split('\n'):
        if line.startswith("ACTION:"):
            result["action"] = line.replace("ACTION:", "").strip()
        elif line.startswith("SEARCH_QUERY:"):
            result["search_query"] = line.replace("SEARCH_QUERY:", "").strip()
        elif line.startswith("MATH_EXPRESSION:"):
            result["math_expression"] = line.replace("MATH_EXPRESSION:", "").strip()

    return result

def ask_agent(question: str, history: list):
    language = detect_language(question)

    routing_text = route_question(question, history)
    routing = parse_routing(routing_text)
    logger.info(f"Routing decision: {routing['action']}")

    context_parts = []

    if routing["action"] in ["SEARCH", "BOTH"]:
        query = routing["search_query"] or question
        search_result = search_jee_material(query)
        context_parts.append(f"Study material:\n{search_result}")

    if routing["action"] in ["CALCULATE", "BOTH"]:
        expr = routing["math_expression"] or ""
        if expr:
            calc_result = calculator(expr)
            context_parts.append(f"Calculation: {calc_result}")

    context = "\n\n".join(context_parts)

    if language == "hi":
        lang_instruction = "You MUST reply in Hindi only."
    else:
        lang_instruction = "You MUST reply in English only. No Hindi whatsoever."

    system_prompt = f"""You are a helpful JEE tutor.
Use the following context to answer the question.
{lang_instruction}
Keep answers concise and JEE-focused.

Context:
{context}"""

    clean_messages = [SystemMessage(content=system_prompt)]

    for msg in history[-6:]:
        content = msg["content"]
        if msg["role"] == "user":
            if detect_language(content) == "hi":
                clean_messages.append(HumanMessage(content=f"[User asked in Hindi]: {translate_to_english(content)}"))
            else:
                clean_messages.append(HumanMessage(content=content))
        else:
            clean_messages.append(SystemMessage(content=f"[Previous answer]: {content[:150]}"))

    answer_language = "Hindi" if language == "hi" else "English"
    clean_messages.append(HumanMessage(content=f"Answer this in {answer_language} only: {question}"))

    response = llm.invoke(clean_messages)
    final_answer = response.content

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": final_answer})

    return final_answer, history
