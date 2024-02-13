from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
import os
from research_assistant import chain as research_assistant_chain
from dotenv import load_dotenv
from langserve.client import RemoteRunnable
app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)

load_dotenv()
os.environ['OPENAI_API_KEY'] = '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'#os.getenv("OPENAI_API_KEY")

add_routes(app, research_assistant_chain, path="/research-assistant")


if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
