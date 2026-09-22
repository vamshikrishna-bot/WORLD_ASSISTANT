from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tools=TavilySearch(max_results=5)
llm= ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt=ChatPromptTemplate.from_template(
    """
You are a helpful assistant

summarize the following news into clear bullet points

{news}
"""
)

chain=prompt | llm | StrOutputParser()

news_result = search_tools.invoke("Latest AI news of 2026")

result=chain.invoke({"news": news_result})

print(result)