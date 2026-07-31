# here we are creating a simple chatbot which remember things until we refresh the flow..
# showcasing how persistence can be used to create a short-term memory. 
# here we are using MemorySaver which saves checkpoints in RAM

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from typing import TypedDict,Annotated

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

# state of chatbot
class ChatBotState(TypedDict):

    messages: Annotated[list[BaseMessage],add_messages]

# chat node
def chat(state: ChatBotState):

    # take user query from state
    messages = state['messages']

    # send to LLM
    response = model.invoke(messages)

    # store response back to state
    return {"messages":[response]}


# creating graph
graph = StateGraph(ChatBotState)

graph.add_node("chat",chat)

graph.add_edge(START,"chat")
graph.add_edge("chat",END)

checkpointer = MemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

# get thread id
thread_id= input("Enter your thread_id: ")

# executing workflow.
while True:
    user_input = input("You: ")

    if user_input.lower().strip() in {"exit","quit","bye"}:
        print("Bot: Nice to talk to you. Have a great day!!")
        break

    # build initial state
    state = {"messages":HumanMessage(content=user_input)}

    # run the graph
    response = workflow.invoke(
        state,
        config={"configurable":{"thread_id":thread_id}}
    )

    # get latest message from bot
    messages = response["messages"]
    last_msg = messages[-1]
    print("Bot: ",last_msg.content)    