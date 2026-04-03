from typing import TypedDict, Sequence, Annotated
import os
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


load_dotenv()


llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

pdf_path = "Project_Catalyst.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")


pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF successfully loaded with {len(pages)}")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise 


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap =200
)


pages_split = text_splitter.split_documents(pages)

persist_directory = f"D:/Projects/Learning/LangraphCourse/Chapter 7"

collection_name = "ProjectCatalyst"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,  
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store")
except Exception as e:
    print(f"Error in creating the DB {str(e)}")
    raise



retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={'k':5} # amount chunks to return
)

@tool
def retriever_tool(query:str)->str:
    """
    This tools searches and returns the information from the document 
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the document"


    results = []

    for i , doc in enumerate(docs):
        results.append(f"Document {i} : {doc.page_content}")
    return "\n\n".join(results)


tools = [retriever_tool]

llm_with_tool = llm.bind_tools(tools=tools)


class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state:AgentState):
    """
    Checks if the last message contains the tool calls

    """

    result = state["messages"][-1]

    tool_calls = getattr(result, "tool_calls", None)
    if tool_calls and len(tool_calls) > 0:
        return True
    return False



system_message = """
You are a retrieval-augmented generation assistant for a PDF knowledge base.
Always use the retriever tool first to answer user questions with grounded document evidence.
If the retriever returns relevant chunks, synthesize a concise, accurate response and cite the source text.
If no relevant information is found, respond with "I found no relevant information in the document" and ask for clarification.
Keep answers focused, factual, and based on the retrieved content.
"""

tools_dict = {our_tool.name : our_tool for our_tool in tools}

def call_llm(state:AgentState):
    """Function to call the llm with current state"""

    messages = list(state["messages"])
    messages = [SystemMessage(content=system_message)] + messages
    message = llm_with_tool.invoke(messages)

    return {"messages" : [message]}


def take_action(state:AgentState):
    """Execute tool calls from the llm's response"""

    tool_calls = getattr(state['messages'][-1], "tool_calls", [])

    results = []

    for tc in tool_calls:
        print(f"Calling tool: {tc['name']} with query :{tc['args'].get('query', 'No query provided')}")

        if not tc['name'] in tools_dict:
            print(f"\n Tool: {tc['name']} does not exists")

            result = "Incorrect Tool Name , PLease Retry and select tool from the list of available tools"

        else:
            result = tools_dict[tc['name']].invoke(tc['args'].get('query', ""))
            print (f"Result length : {len(str(result))}")

        

        results.append(ToolMessage(tool_call_id = tc['id'], name = tc['name'], content= str(result)))

    
    print("Tool Execution Complete , Back to the model")

    return {'messages':results}



graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True:"retriever_agent",
        False:END
    }
)

graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")


agent = graph.compile()


def running_rag():
    print(f"Welcome to Document Helper \n\n")

    while True:
        user_input = input("What would you like to know about the document?: ")
        if user_input.lower() in ['exit', "quit"]:
            break
            
        messages = [HumanMessage(content=user_input)]

        result = agent.invoke({'messages' : messages})

        print("Answer: \n")
        print(result['messages'][-1].content)


running_rag()