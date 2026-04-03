from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


load_dotenv()

llm = init_chat_model(model="gpt-5-mini", temperature=0)

document_content = ""

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


@tool
def update_tool(content:str) -> str:
    """
    Updates the document with the provided content
    """

    global document_content

    document_content = content
    return f"Document has been updated successfully and the current content is \n{document_content}"

@tool
def save_tool(filename:str) ->str:
    """
    Save the current document to a text file and finish the process

    Args:
        filename: Name for the text file

    """

    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
            return f"Document has been save successfully to {filename}"
    
    except Exception as e:
        return f"Error saving document:{str(e)}"
    

tools = [update_tool, save_tool]

llm_with_tools = llm.bind_tools(tools=tools)

def our_agent(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant , You are going to help the user to update and modify the documents.
                                  
     - If the users want to update or modify the content , use the 'update' tool with the completed updated content
     - If the user wants the save and finish , you need to use the 'save' tool.
     - Make sure to always show the current document state after modifications

    The current document content is :{document_content}                            
    """
    )

    if not state['messages']:
        user_input = "Im ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\n What would you like to do the document?")
        print(f"\n User: {user_input}")
        user_message = HumanMessage(content=user_input)
    
    all_messages = [system_prompt] + list(state['messages']) + [user_message]
    response = llm_with_tools.invoke(all_messages)

    print(f"/n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"Using Tools: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages" : list(state['messages']) + [user_message, response]}


def should_continue(state:AgentState)->str:
    """
    Determine if we should contine or end the conversation
    """

    messages = state['messages']

    for message in messages:
        if (isinstance(message, ToolMessage) and 
            isinstance(message.content, str) and
            "saved" in message.content.lower() and 
            "document" in message.content.lower()) :
            return END

    return "continue"



def print_message(messages):
    """
    Funtion to print message in to console in a more readble format
    """


    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n Tool Result: {message.content}")
 

graph = StateGraph(AgentState)
graph.add_node("our_agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "our_agent")
graph.add_edge("our_agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue":"our_agent",
        "end":END
    }
)

app = graph.compile()


def run_document_agent():
    """
    This method runs the agent
    """

    state: AgentState = {"messages" : []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_message(step['messages'])
        
    print("\n Drafter Finished")


if __name__ == "__main__":
    run_document_agent()