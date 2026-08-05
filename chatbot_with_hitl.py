from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langchain_community.tools import tool
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import requests

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# tools
@tool
def get_stock_price(symbol:str):

    """
    Fetch latest stock price for given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=8NP3PSK04VUM3WAO"
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol:str, quantity:int):

    """
    simulate purchasing a given quantity of a stock symbol

    HUMAN-IN-THE-LOOP:
    before confirming the purchase, this tool will interupt.
    and wait for human decision (yes / anything else)
    """

    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes / no)")

    if isinstance(decision,str) and decision.lower() == "yes":
        return {
            "status":"success",
            "message":f"Purchase order placed for {quantity} shares of {symbol}.. :)",
            "symbol":symbol,
            "quantity":quantity
        }
    else:
        return {
                    "status":"cancelled",
                    "message":f"Purchase order for {quantity} shares of {symbol} cancelled by user.. :(",
                    "symbol":symbol,
                    "quantity":quantity
                }

tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)

# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    """LLM that may answer or call a tool """
    message = state["messages"]
    response = llm_with_tools.invoke(message)
    return {"messages":[response]}

tool_node = ToolNode(tools)

checkpointer = MemorySaver()

# create graph
graph = StateGraph(ChatState)

graph.add_node("chat",chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"chat")
graph.add_conditional_edges("chat",tools_condition)
graph.add_edge("tools","chat")

chatbot = graph.compile(checkpointer=checkpointer)

# Simple usage example (CLI with HITL)

if __name__ == "__main__":

    thread_id = "trail-1"

    while True:
        user_input = input("you: ")
        if user_input.lower().strip() in {"exit","quit","bye"}:
            print("AI: Goodbye....!!")
            break

        state = {"messages":[HumanMessage(content=user_input)]}

        result = chatbot.invoke(
            state,
            config={"configurable":{"thread_id":thread_id}}
        )

        interrupts = result.get("__interrupt__",[])

        if interrupts:

            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()


            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable":{"thread_id":thread_id}}
            )

        # get latest message
        messages = result["messages"]
        last_msg = messages[-1]
        print(f"AI: {last_msg.content}\n")
