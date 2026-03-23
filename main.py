import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent import ask_agent



app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)


sessions = {}  #Store each user chat history seperately, session_id-value


#What request will look like
class ChatRequest(BaseModel):
    question:str
    session_id:str = "default"



class ChatResponse(BaseModel):
    answer: str
    session_id : str


@app.get("/")
def root():
    return {"status": "AI JEE agent is running"}


@app.post("/chat", response_model = ChatResponse)
def chat(request: ChatRequest):

    if request.session_id not in sessions:
        sessions[request.session_id] = []

    history = sessions[request.session_id]

    answer, updated_history = ask_agent(request.question, history)

    sessions[request.session_id] = updated_history

    return ChatResponse(answer=answer, session_id=request.session_id)



@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return{"Cleared": session_id}


