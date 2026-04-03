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

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int):
    """
    This is addition function which adds a and b and returns the result
    """
    return a + b

@tool
def mul(a:int, b:int):
    """
    This is multiplies function which multiplies a and b and returns the result
    """
    return a * b

tools = [add, mul]

llm_with_tools = llm.bind_tools(tools=tools)


def model_call(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content="You are my AI Assistant, please answer my queries to the best of your ability")
    response = llm_with_tools.invoke([system_prompt] + list(state['messages']))

    return {'messages' : [response]}


def should_continue(state:AgentState)->str:
    messages = state['messages']
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if not tool_calls:
        return "exit"
    else:
        return "continue"
    


graph = StateGraph(AgentState)

graph.add_node("model_call", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)


graph.set_entry_point("model_call")

graph.add_conditional_edges(
    "model_call",
    should_continue,
    {
        "continue":"tools",
        "exit":END
    }
)

graph.add_edge("tools", "model_call")

agent = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
    
inputs: AgentState = {
    "messages": [HumanMessage(content="Add 20 + 20 and then multiply the result by 6 and also tell me a joke please")]
}

print_stream(agent.stream(inputs, stream_mode="values"))