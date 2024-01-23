from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain.agents import load_tools, ZeroShotAgent, AgentExecutor
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.agent_toolkits import FileManagementToolkit

from funcs import get_input
from c_tools import (
    nothing,
    image_information_search,
    new_face_addition,
    facial_recognition,
    add_important_context,
)
from os import system, environ

llm = Ollama(
    model="mistral:instruct",
    verbose=True,
    temperature=0.2,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

tools = load_tools(["human", "ddg-search", "google-search"], llm=llm, input_func=get_input)
file_tools = FileManagementToolkit(
    root_dir=str(""),
    selected_tools=["read_file", "list_directory"],
).get_tools()

tools = (
    tools
    + file_tools
    + [
        nothing,
        image_information_search,
        new_face_addition,
        facial_recognition,
        add_important_context,
    ]
)

prefix = """You are Yashu's assistant Jarvis on Mac with root directory at __root_dir__; 
Your location is Lucknow, India. prioritize Yashu's wishes, avoid human queries, explore web/files first for information.
Always use the dedicated image tools for image files first. ALWAYS USE THE ACTION/ACTION INPUT SYNTAX. 
if you are trying to perform some action then it is mandatory for you to give an action input as an Action Input. 
Always reiterate on input or command given by Yashu to always get the relevant answer or action.
Add the context using the appropriate tool and use it if you think that the information observed might come in use in the future. 
Always pass in some kind of Action. If you want to give any information to Yashu, pass it as the Final Answer after analysing it. 
Don't use the human tool for giving information. Always use the appropriate tool for the given task.
Give short and precise answers. Don't give long answers.
When you want to give some information to Yashu, give it as the Final Answer only.
Always break your problem into multiple smaller parts and then try to solve them using the given tools and reiterate on the given command and your observations before moving to the next Action.
Always divide your problem into smaller parts and then try to solve them using the given tools and reiterate on the given command and your observations before moving to the next Action.
Sometimes you might not need to use a tool and you can directly give the answer as the Final Answer."""

suffix = """you now will be talking directly with Yashu. Don't ask unecessary questions from Yashu or human; Begin! 
{context}

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad", "context"],
)
memory = ConversationBufferWindowMemory(
    k=5, memory_key="chat_history", input_key="input", return_messages=True
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True, memory=memory)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors="""Try passing some Action Input to the action. Make sure to give input as the Action Input only and nothing else. If you know the answer, return the Final Answer.""",
    max_iterations=25,
)


def main(command):
    context = ""
    with open("context.txt", "r") as f:
        context = f.read()
        out = agent_chain.invoke({"input": command, "context": context})
    print(out)
    system(f"""say "{out["output"]}" """)


if __name__ == "__main__":
    while True:
        command = input(">>> ")
        if command == "exit":
            break
        main(command)
