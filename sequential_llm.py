from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

os.environ['LANGCHAIN_PROJECT']='Sequential_Langchain'

llm1=ChatGroq(model='llama-3.1-8b-instant')
llm2=ChatGroq(model='openai/gpt-oss-20b')

parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Generate a report on topic :{topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Based on the given report give 5 key points : {text}",
    input_variables=['text']
)

chain=prompt1 | llm1 | parser | prompt2 | llm2 | parser

response=chain.invoke({'topic':'Education In India'},cofig={'run_name':'sequential_chain'})

print(response)