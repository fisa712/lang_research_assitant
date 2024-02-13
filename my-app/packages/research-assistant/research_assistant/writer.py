from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
import os

WRITER_SYSTEM_PROMPT = "You are an AI News Researcher and News Expander. Your sole purpose is to analyze the provided news content and rephrase it to a news article tone. You should write well written, critically acclaimed factual news content, objective and structured news articles on given news text."  # noqa: E501


# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py

RESEARCH_REPORT_TEMPLATE = """Information: 
--------
{research_summary}
--------

Using the above information,  you have to rephrase it into a proper news article tone and make sure you dont miss any information in the provided news content -- \
The news article should focus on the answer to the topic, should be well structured, informative, \

You should rephrase to write the news article as long as you can using all relevant and necessary information provided. The minimum length generated news article can have is 1,500 words to 2,000 words.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, to generate a detailed and lengthy news article."""  # noqa: E501


# RESOURCE_REPORT_TEMPLATE = """Information:
# --------
# {research_summary}
# --------
#
# Based on the above information, generate a bibliography recommendation report for the following question or topic: "{question}". \
# The report should provide a detailed analysis of each recommended resource, explaining how each source can contribute to finding answers to the research question. \
# Focus on the relevance, reliability, and significance of each source. \
# Ensure that the report is well-structured, informative, in-depth, and follows Markdown syntax. \
# Include relevant facts, figures, and numbers whenever available. \
# The report should have a minimum length of 1,200 words.
#
# Please do your best, this is very important to my career."""  # noqa: E501

# OUTLINE_REPORT_TEMPLATE = """Information:
# --------
# {research_summary}
# --------
#
# Using the above information, generate an outline for a research report in Markdown syntax for the following question or topic: "{question}". \
# The outline should provide a well-structured framework for the research report, including the main sections, subsections, and key points to be covered. \
# The research report should be detailed, informative, in-depth, and a minimum of 1,200 words. \
# Use appropriate Markdown syntax to format the outline and ensure readability.
#
# Please do your best, this is very important to my career."""  # noqa: E501
os.environ['OPENAI_API_KEY'] = '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'
model = ChatOpenAI(model='gpt-4-0125-preview', temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)
# ).configurable_alternatives(
#     ConfigurableField("report_type"),
#     default_key="research_report",
#     resource_report=ChatPromptTemplate.from_messages(
#         [
#             ("system", WRITER_SYSTEM_PROMPT),
#             ("user", RESOURCE_REPORT_TEMPLATE),
#         ]
#     ),
#     outline_report=ChatPromptTemplate.from_messages(
#         [
#             ("system", WRITER_SYSTEM_PROMPT),
#             ("user", OUTLINE_REPORT_TEMPLATE),
#         ]
#     ),
# )
chain = prompt | model | StrOutputParser()


# answer the following question or topic: "{question}"
#in depth analyzed, with facts and numbers if available and of at least 1,200 to 1,500 words long to make it a detailed news article.
