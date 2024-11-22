
# %%
from dotenv import load_dotenv
import os
load_dotenv()
os.environ['LANGCHAIN_PROJECT'] ="langgraph-chatbot"


from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

# llm = ChatOllama(model="llama3.2")
llm = ChatGroq(model="llama-3.2-90b-vision-preview")

from pydantic import BaseModel

class PromptInput(BaseModel):
  """Prompt generatiion requirements."""
  objective: str
  variables: list[str]
  constraints: list[str]
  requirements: list[str]
  
def convert_to_prompt_input(objective: str, variables: list[str], constraints: list[str], requirements: list[str]):
  """This is a tool which takes in reeusted prompt's objective, variables, constraints, and requirements and instantiate PromptInput object

  Args:
      objective: str; objective of the LLM assistant prompt
      variables: list of str; variables in the prompts that will be substituted later
      constraints: list of str; any constraints for the AI assistant
      requirements: list of str; any requrements for the AI assistant

  Raises:
      ValueError: _description_

  Returns:
      _type_: return in dictionary format
  """
  requirements = {
    "objective": objective,
    "variables": variables,
    "constraints": constraints,
    "requirements": requirements
  }

  return requirements
  
class State(MessagesState):
  requirements: PromptInput = None
  human_feedback: str
  error: str
  pass



# Information Gathering Node
info_gatherings = [convert_to_prompt_input]
llm_with_gathering = llm.bind_tools(info_gatherings)
# llm_with_gathering = llm.with_structured_output(PromptInput)

info_sys_prompt = """
Your job is to get information from the user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.


After you are able to discern all the information, call the relevant tool.
"""

def info_gathering_llm(state):
  response = llm_with_gathering.invoke([SystemMessage(info_sys_prompt)] + state['messages'])
  new_state = {
    'messages': response  
  }
  return new_state

## Prompt generating node
  
import json

def prompt_input_populator(state):
  # previous node must be the Tool
  # assert isinstance(tool_message := state['messages'][-1], ToolMessage)
  requirements_formt = {
    "objective": "string",
    "variables": ["string"],
    "constraints": ["string"],
    "requirements": ["string"]
  }
  sys_msg = f"""
  I want you to convert the following string into a JSON object. If the information is not in the expected format, please return with default values.
  This is not a code. I want you to only provide a JSON object in the following format:
  If any of the information is not provide, give me the json object with all the keys but empty values.
  
  This is the structure of the JSON Object: {requirements_formt}
  
  REMEMBER! You MUST ONLY return the json object.
  """
  result = llm.invoke([SystemMessage(content=sys_msg), state['messages'][-1]] )
  
  return {
    "requirements": result.content
  }
  


def prompt_generating_llm(state):
  prompt_sys_msg = f"""
  Using the following requirements, write a good prompt template

  requirements: {state['requirements']}
  """
  response = llm.invoke([SystemMessage(content=prompt_sys_msg)] + state['messages'])
  
  return {
    'messages': response
  }
  
def additional_human_info(state):
  pass
  
# def info_gathering_conditional(state):
#   # When the all the requirements were acquired by the user and the input is generated
#   if isinstance(last_message := state['messages'][-1], AIMessage) and state.get('requirements'):
#     print('prompt call')
#     return "prompt_llm"
    
#   # Information from the human input was insufficient, and need to get additional info from human
#   else:
#     print("does it ever come here?")
#     return "info_llm"
  
  
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from sqlite3 import connect
from typing import Literal

def info_tools_condition(state) -> Literal["info_tools", "additional_human_info"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get("messages", [])):
        ai_message = messages[-1]
    elif messages := getattr(state, "messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "info_tools"
    return "additional_human_info"

builder = StateGraph(State)
builder.add_node('info_llm', info_gathering_llm)
builder.add_node('prompt_populator', prompt_input_populator)
builder.add_node(ToolNode(info_gatherings, name="info_tools"))
builder.add_node(additional_human_info)
builder.add_node('prompt_llm', prompt_generating_llm)


builder.add_edge(START, "info_llm")
builder.add_conditional_edges('info_llm', info_tools_condition)
builder.add_edge('additional_human_info', 'info_llm')
builder.add_edge('info_tools', 'prompt_populator')
builder.add_edge('prompt_populator', 'prompt_llm')
# builder.add_edge('info_tools', 'prompt_llm')
builder.add_edge('prompt_llm', END)

# db_path = "db/prompt_gen.db"
# conn = connect(db_path)
# memory = SqliteSaver(conn)

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

graph = builder.compile(checkpointer=memory, interrupt_before=['additional_human_info'])

#  %%
from IPython.display import display, Image
display(Image(graph.get_graph().draw_mermaid_png()))


#%%
state = {
  'messages': [HumanMessage(input("AI: What kind of prompt would you like me to generate?:\n"))]
}

config = {
  'configurable': {'thread_id': 1}
}
res_state = graph.invoke(state, config=config)   # change to astream_events
for m in res_state['messages']:
  m.pretty_print()

# %%

# for snapshot in list(graph.get_state_history(config)):
#   print(snapshot)

# list(graph.get_state_history(config))[0].values['messages'][-1]


# %%
while True:
  if not graph.get_state(config).next or not graph.get_state(config).next[0] == "additional_human_info":
    break
  user_input = input("Please add any feedbacks: \n")
  updated_state = {
    "messages": [HumanMessage(content=user_input)]
  }
  graph.update_state(config, updated_state)

  res = graph.invoke(None, config=config)
  for m in res['messages']:
    m.pretty_print()

# I want you to generate a prompt for a arithmatic ai assisant
# 1. all of them, 2. all of them, 3. no constraints, 4. no requirements

snapshot_history = list(graph.get_state_history(config))
# # %%
# res = graph.invoke(None, config=config)
# res['messages'][-1].pretty_print()

# # %%
# list(graph.get_state_history(config))[0].next

