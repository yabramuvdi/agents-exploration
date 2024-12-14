# Resources:
# - https://huggingface.co/docs/transformers/main/agents_advanced
# - https://huggingface.co/docs/transformers/agents

#%%

# installations
# pip install duckduckgo-search markdownify

#%%

from transformers import HfApiEngine
from transformers.agents import ReactCodeAgent, ReactJsonAgent, HfApiEngine, ManagedAgent
from transformers.agents.search import DuckDuckGoSearchTool, VisitWebpageTool
from huggingface_hub import login, InferenceClient

# authenticate
with open("/Users/yabra/keys/hf_key.txt") as f:
    hf_key = f.read()

login(hf_key)
# %%

#=============
# Define agents
# Agent 1: Extract the song and artist of a song
# Agent 2: Performs a web search for the lyrics of the song
# Agent 3: Extracts the lyrics from the webpage
#=============

llm_model_path = "meta-llama/Meta-Llama-3-70B-Instruct"
llm_engine = HfApiEngine(model=llm_model_path)

# Agent 1
parse_input_agent = ReactJsonAgent(tools=[], llm_engine=llm_engine)
managed_parse_input_agent = ManagedAgent(
    agent=parse_input_agent,
    name="parse_input",
    description="Extracts the name of the song and the artist from the user input. If no artist is mention then it performs its best guess."
)

# Agent 2
find_lyrics_site_agent = ReactCodeAgent(tools=[DuckDuckGoSearchTool()], 
                                        llm_engine=llm_engine,
                                        additional_authorized_imports=['requests'])
managed_find_lyrics_site_agent = ManagedAgent(
    agent=find_lyrics_site_agent ,
    name="find_lyrics_site",
    description="Runs web searches to find the lyrics of a song. Give it your query as an argument. Returns the best website for the required lyrics."
)


# Agent 3
web_parsing_agent = ReactCodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool()], 
                                   llm_engine=llm_engine,
                                   additional_authorized_imports=['requests', 'bs4']
                                   )

managed_web_parsing_agent = ManagedAgent(
    agent=web_parsing_agent,
    name="web_parsing_lyrics",
    description="Visits the website provided (DO NOT MODIFY THE URL) and retrives lyrics from the song. Returns the extracted text. If it is not possible to clean the text, just return all the content of the webpage."
)

# # Agent 4
# lyrics_cleaning_agent = ReactJsonAgent(tools=[], llm_engine=llm_engine)

# managed_lyrics_cleaning_agent = ManagedAgent(
#     agent=lyrics_cleaning_agent,
#     name="lyrics_cleaning",
#     description="Cleans the extracted text from the lyrics website to only contain the text from the lyrics of the relevant song. Returns the result in markdwon."
# )

#%%

#=======
# Define a manager
#=======

manager_agent = ReactCodeAgent(
    tools=[], 
    llm_engine=llm_engine, 
    managed_agents=[managed_parse_input_agent, 
                    managed_find_lyrics_site_agent, 
                    managed_web_parsing_agent,
                    #managed_lyrics_cleaning_agent,
                    ]
)

#%%

task = "Get the lyrics to hips dont lie"
max_attempts = 5
done = False
attempts = 0
while not done and attempts < max_attempts:
    result = manager_agent.run(task)
    try:
        final_lyrics = result["Task outcome (extremely detailed version)"]
        done = True
    except Exception as e:
        print(f"{e}")
        print("Retrying")
        attempts += 1

print(final_lyrics)

    
# %%
