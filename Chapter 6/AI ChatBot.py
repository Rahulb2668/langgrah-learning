import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph,START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages:List[Union[HumanMessage, AIMessage]]

llm = init_chat_model(model="gpt-5-mini", temperature=0)


def process(state:AgentState)->AgentState:
    """ This Node will process the messages with AI """

    response = llm.invoke(state['messages'])

    state['messages'].append(AIMessage(content=response.content))
    print(f"\n AI: {response.content}")

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.set_entry_point("process")
graph.set_finish_point("process")

agent = graph.compile()

# Load the previous history

conversation_history = []
    
log_file = "AIChatBot1_Log.txt"
if os.path.exists(log_file):
    with open(log_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("You: "):
                conversation_history.append(HumanMessage(content=line.replace("You: ", "").strip()))
            elif line.startswith("AI: "):
                conversation_history.append(AIMessage(content=line.replace("AI: ", "").strip()))
    print("--- Previous conversation loaded ---")


user_input = input("You: ")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages":conversation_history})

    conversation_history = result['messages']

    user_input = input("You: ")


with open("AIChatBot1_Log.txt", "a", encoding="utf-8") as file:
    file.write("Your Conversation History\n")

    for conversation in conversation_history:
        if isinstance(conversation, HumanMessage):
            file.write(f"You: {conversation.content}\n")
        elif isinstance(conversation, AIMessage):
            file.write(f"AI: {conversation.content}\n")
    file.write("End of Conversation")
    
    print("Your Conversation is saved to AIChatBot1_Log.txt")