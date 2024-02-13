import json
import os
from typing import Any

import requests
from bs4 import BeautifulSoup
from langchain.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities import GoogleSerperAPIWrapper

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

RESULTS_PER_QUESTION = 2

# os.environ['SERPAPI_API_KEY'] = '2a0633fb51743ca2b9360b6662dcb53267be8aaf15aa5b7d063df899c2115ef8'

# ddg_search = DuckDuckGoSearchAPIWrapper()
# serp_search = SerpAPIWrapper(params={'tbm': 'nws', 'tbs': 'qdr:m', 'num': 2})
os.environ["SERPER_API_KEY"] = "077bdf1a3b5cbb61187262eb2439b243184ec6cb"
search = GoogleSerperAPIWrapper(type='news')
os.environ['TAVILY_API_KEY'] = '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'
def scrape_text(url: str):
    # Send a GET request to the webpage
    try:
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)

            # Print the extracted text
            # print("====================")
            # print("url: ", url)
            # print("The extracted text: ", page_text)
            # print("====================")
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)  # noqa: T201
        return f"Failed to retrieve the webpage: {e}"


def web_search(query: str, num_results: int):
    # results = ddg_search.results(query, num_results)
    # results = serp_search.run(query)
    results = search.run(query)
    print("Serper results: ", results)
    return [r["link"] for r in results]


get_links: Runnable[Any, Any] = (
    RunnablePassthrough()
    | RunnableLambda(
        lambda x: [
            {"url": url, "question": x["question"]}
            for url in web_search(query=x["question"], num_results=RESULTS_PER_QUESTION)
        ]
    )

).configurable_alternatives(
    ConfigurableField("search_engine"),
    default_key="serpAPI",
    tavily=RunnableLambda(lambda x: x["question"])
    | RunnableParallel(
        {
            "question": RunnablePassthrough(),
            "results": TavilySearchAPIRetriever(k=RESULTS_PER_QUESTION),
        }
    )
    | RunnableLambda(
        lambda x: [
            {"url": result.metadata["source"], "question": x["question"]}
            for result in x["results"]
        ]
    ),
)


SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{agent_prompt}"),
        (
            "user",
            "Write 5 standalone google search queries that must contain respective key words to search online that "
            "form an objective opinion from the following news related data : {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3", "query 4", "query 5"].',
        ),
    ]
)

AUTO_AGENT_INSTRUCTIONS = """
This task involves researching a given news topic, regardless of its complexity or the availability of a definitive answer. The research is conducted by a an agent, defined by its type and role, with agent requiring distinct instructions.
Agent
The agent is only determined by news field and the specific name of the agent as News Agent or Researcher that could be utilized to research the topic provided. News Agent is known for its area of expertise, and agent type is associated with a corresponding emoji.

examples:
task: "how industrial development impacting the economy of developing countries?"
response:
{
    "agent": " 📰 News Agent",
    "agent_role_prompt: "You are a skilled news generator AI assistant. Your primary goal is to compose comprehensive, detailed, impartial, and methodically arranged news articles based on provided news data."
}

"""  # noqa: E501
CHOOSE_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [SystemMessage(content=AUTO_AGENT_INSTRUCTIONS), ("user", "task: {task}")]
)

SUMMARY_TEMPLATE = """{text} 

-----------

Using the above text, rephrase the whole text in detailed way to answer the following question: 

> {question}
 
-----------
Try to keep the the focus on the question while rephrasing the news text,if the question cannot be focused for the text, simply rephrase the text without missing any relevant information. Include all factual information, numbers, stats, links etc if available."""  # noqa: E501

SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

scrape_and_summarize: Runnable[Any, Any] = (
    RunnableParallel(
        {
            "question": lambda x: x["question"],
            "text": lambda x: scrape_text(x["url"])[:10000],
            "url": lambda x: x["url"],
        }
    )
    | RunnableParallel(
        {
            "summary": SUMMARY_PROMPT | ChatOpenAI(model='gpt-4-0125-preview', temperature=0, openai_api_key = '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') | StrOutputParser(),
            "url": lambda x: x["url"],
        }
    )
    | RunnableLambda(lambda x: f"Source Url: {x['url']}\nSummary: {x['summary']}")
)

multi_search = get_links | scrape_and_summarize.map() | (lambda x: "\n".join(x))


def load_json(s):
    try:
        return json.loads(s)
    except Exception:
        return {}


search_query = SEARCH_PROMPT | ChatOpenAI(model='gpt-4-0125-preview', temperature=0, openai_api_key = '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') | StrOutputParser() | load_json
choose_agent = (
    CHOOSE_AGENT_PROMPT | ChatOpenAI(model='gpt-4-0125-preview', temperature=0, openai_api_key = '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') | StrOutputParser() | load_json
)

get_search_queries = (
    RunnablePassthrough().assign(
        agent_prompt=RunnableParallel({"task": lambda x: x})
        | choose_agent
        | (lambda x: x.get("agent_role_prompt"))
    )
    | search_query
)
# print("Search Queries: ", get_search_queries)

chain = (
    get_search_queries
    | (lambda x: [{"question": q} for q in x])
    | multi_search.map()
    | (lambda x: "\n\n".join(x))
)

# print('chain: ', chain)
