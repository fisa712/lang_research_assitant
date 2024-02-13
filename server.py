import os
from research_assistant import chain as research_assistant_chain
from dotenv import load_dotenv
from langserve.client import RemoteRunnable
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

add_routes(app, research_assistant_chain, path="\research-assistant")

runnable = RemoteRunnable("http://localhost:8000/research-assistant")
# app = FastAPI(title="lang_researcher")

