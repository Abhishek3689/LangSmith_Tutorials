from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

# os.getenv('GROQ_API_KEY')
load_dotenv()

llm=ChatGroq(model='llama-3.1-8b-instant')

prompt=PromptTemplate(
    template="based on topic give a report not more than 300 tokens/words: {topic}",
    input_variables=['topic']
)

parser=StrOutputParser()

chain=prompt | llm | parser

res=chain.invoke({'topic':'Education system in India'})

print(res)