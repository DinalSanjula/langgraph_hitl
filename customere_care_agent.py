import operator

from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph

load_dotenv()

model = ChatOpenAI(name='gpt-5.2-2025-12-11')

@tool
def check_order_status(order_id:str):
    """Check the delivery  status of an order"""

    return f"Order {order_id} is currently out for delivery"
@tool
def issue_refund(order_id: str, amount: float):
    """Issue a refund for an order"""
    return f"Refund of ${amount} has been initiated for order {order_id}"
@tool
def escalate_to_human(ticket_id:str):
    """Escalate a ticket to a human support agent"""
    return f"Ticket {ticket_id} has been escalated to a human agent"

tools = [check_order_status, issue_refund, escalate_to_human]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    approve: bool

def planning_node(state: MessagesState):
    response = model_with_tools.invoke(

        [SystemMessage(content=

                       """
                       You are an customer support assistant that propose which tool should be used ,
                       but you MUST NOT call tools yet.
               
                       Respond with a clear plan like:
                           - Tool: check_order_status(order_id=123)
                           - Reason: Customer is asking about delivery
                       """)] + state["messages"]
    )

    return {"messages": [response], "approve": False}

def human_review_node(state: MessagesState):
    print("=====Proposed Tool Plan Bellow!!!!=====")

    print(state["messages"][-1].content)

    decision = input("Enter your decision ? approved/ escalte to agent ")

    if decision == "approved":
        return {"approve": True}

    return {
        "messages" : [HumanMessage(content=f"Modify the plan as follow : {decision}")],#langgraph framework appends the new message to the previous message list instead of replacing.
        "approve" : False
    }

def execute_tool_node(state: MessagesState):
    response = model_with_tools.invoke([SystemMessage(content="""You can now execute the approved plans,
                                        using available tools""")]+state["messages"])

    return {"messages": [response]}

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def output_llm_node(state:MessagesState):

    response = model_with_tools.invoke([SystemMessage(content="""Provide the Final answer to user with final tool call output""")]+state["messages"])

    return {"messages": [response]}

def check_human_review(state:MessagesState):

    if state["approve"] :
        return  "execute_tool_node"

    else:
        return "human_review"

def check_tool_calls(state: MessagesState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool_node"

    return END

graph = StateGraph(MessagesState)

graph.add_node("planning_node",planning_node)
graph.add_node("human_review", human_review_node)
graph.add_node("execute_tool_node", execute_tool_node)
graph.add_node("tool_node",tool_node)
graph.add_node("output_llm_node",output_llm_node)

graph.add_edge(START,"planning_node")
graph.add_edge("planning_node","human_review")
graph.add_conditional_edges("human_review", check_human_review,["execute_tool_node","human_review"])
graph.add_conditional_edges("execute_tool_node",check_tool_calls,["tool_node","output_llm_node"])

graph.add_edge("tool_node","output_llm_node")
graph.add_edge("output_llm_node",END)

agent = graph.compile()

while 1:
    question = input("Enter your question ")
    messages = [HumanMessage(content=question)]
    messages = agent.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
