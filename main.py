from openai import OpenAI
import os
import sys
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

base_url = os.environ.get("OPENAI_BASE_URL") 
api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    embedding = OpenAIEmbeddings(base_url=base_url)
    vector_store = Chroma.from_documents(document_chunks, embedding)
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(base_url=base_url)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(base_url=base_url)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# TODO: handle arguments
input_url = sys.argv[1]
vector_store = get_vectorstore_from_url(input_url)

print("AI: Hello, do you have any questions about this website")
chat_history = [
    AIMessage(content="Hello, do you have any questions about this website")
]

# TODO: use stream
while True:
    user_question = input("You: ")
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    res = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_question
    })
    print("AI: ", res["answer"])
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=res["answer"]))
