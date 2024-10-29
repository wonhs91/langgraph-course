from dotenv import load_dotenv
from langchain_ollama import ChatOllama


load_dotenv()

llm = ChatOllama(model="llama3.2")

# Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [wiki_tool]

llm_with_wiki = llm.bind_tools(tools)


# State

from langgraph.graph import MessagesState, StateGraph, START, END
from typing import Annotated

def increment_num(orig, new):
  if orig == None:
    return 0
  return orig + 1


class State(MessagesState):
  track_num: Annotated[int, increment_num]
  pass

# Node
from copy import deepcopy
from langchain_core.messages import SystemMessage, HumanMessage

def wiki_calling_llm(state):
  new_state = deepcopy(state)
  sys_msg = SystemMessage(content="You are a research assitant that will help find answer to the user's questions.")
  new_state["messages"] = [llm_with_wiki.invoke([sys_msg] + state["messages"])]
  
  return new_state


# Graph builder
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(State)

builder.add_node('wiki_llm', wiki_calling_llm)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, 'wiki_llm')
builder.add_conditional_edges('wiki_llm', tools_condition)
builder.add_edge('tools', 'wiki_llm')

memory = MemorySaver()

graph = builder.compile(memory)

