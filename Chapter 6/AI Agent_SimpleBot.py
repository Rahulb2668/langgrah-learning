from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph,START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages:List[HumanMessage]


llm = init_chat_model(model="gpt-5-mini", temperature=0)


def process(state:AgentState)->AgentState:
    response = llm.invoke(state['messages'])
    print(f"\n AI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)

graph.set_entry_point("process")
graph.set_finish_point("process")

agent = graph.compile()

user_input = input("Enter")

while user_input != "exit":
    agent.invoke({"messages":[HumanMessage(content=user_input)]})
    user_input = input("Enter: j")
