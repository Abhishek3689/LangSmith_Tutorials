from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langsmith import traceable
import os,time

start=time.time()
load_dotenv()

os.environ['LANGCHAIN_PROJECT']='RAG_Version_1.0'

## embedding and LLm models
embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
llm=ChatGroq(model='groq/compound')
parser=StrOutputParser()

artifacts_folder='artifacts'
os.makedirs(artifacts_folder,exist_ok=True)

index_path=os.path.join(artifacts_folder,'faiss_index')
# Document path
path='machine_learning_1.pdf'

## Document Loader
@traceable(name='load_pdf')
def load_pdf(path:str):
    loader=PyPDFLoader(path)
    docs=loader.load()
    return docs

## Chunks Creation
@traceable(name='splitter')
def split_docs(docs):
    splitter=RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=150)
    chunks=splitter.split_documents(docs)
    return chunks

## Knowledge Base
@traceable(name='vector_store')
def vector_store_base(chunks):
    if os.path.exists(index_path):
        print("Loading from Exisiting Faiss Index")
        vs=FAISS.load_local(index_path,embeddings=embedding_model,allow_dangerous_deserialization=True)
    else:
        print("Creating New Faiss Index")
        vs=FAISS.from_documents(documents=chunks,embedding=embedding_model)
        vs.save_local(index_path)
    return vs

@traceable(name='setup_pipeline')
def setup_pipeline(path:str):
    docs=load_pdf(path)
    chunks=split_docs(docs)
    vs=vector_store_base(chunks)
    return vs

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

vs=setup_pipeline(path)
retriever=vs.as_retriever(search_type='similarity',search_kwargs={"k":4})

prompt=PromptTemplate(
    template="""Answer the following question based on the provided context only. 
    I want you to be strict if answer not present in context , simply say answer not in relevant context \n
    question:{user_question}\n
    context:\n{context}""",
    input_variables=['user_question',"context"]
)


## parallel Chain
parallel_chain=RunnableParallel(
    {
        "user_question": RunnablePassthrough(),
        "context": retriever | RunnableLambda(format_docs)
    }
)

chain=parallel_chain | prompt | llm | parser


config = {
    "run_name": "pdf_rag_query"
}
# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q=input("Ask Question: ")
ans=chain.invoke(q.strip(),config=config)
print("\nA:",ans)
print("*"*100)
print("Time Taken :",time.time()-start)