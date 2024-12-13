
#%%

from transformers import HfApiEngine
from transformers.agents import ReactCodeAgent, ReactJsonAgent, HfApiEngine, DuckDuckGoSearchTool, ManagedAgent, VisitWebpageTool
from huggingface_hub import login, InferenceClient

data_path = "../data/"

# authenticate
with open("/Users/yabra/keys/hf_key.txt") as f:
    hf_key = f.read()

login(hf_key)

#%%

#=============
# Define agents
# Agent 1: Opens dataset, gets column names and basic descriptive statistics of variables
# Agent 2: Formulates several possible interesting questions from the data
# Agent 3: Writes code to implement each proposed analysis
# Agent 4: Formulates conclusions based on the results of the analysis
#=============

reasoning_model_path = "meta-llama/Meta-Llama-3-70B-Instruct"
coding_model_path = "Qwen/Qwen2.5-72B-Instruct"

# Agent 1
data_exploration_agent = ReactCodeAgent(
    tools=[],
    llm_engine=coding_model_path,
    additional_authorized_imports=["numpy", "pandas", "matplotlib.pyplot", "seaborn"],
    max_iterations=10,
)
managed_agent1 = ManagedAgent(
    agent=data_exploration_agent,
    name="data_exploration",
    description="Opens dataset, gets column names and basic descriptive statistics of variables"
)

# Agent 2
hypothesis_formulation_agent = ReactJsonAgent(
    tools=[],
    llm_engine=reasoning_model_path,
)
managed_agent2 = ManagedAgent(
    agent=hypothesis_formulation_agent,
    name="hypothesis_formulation",
    description="Formulations several interesting hypothesis that could be explored from the given data."
)

# Agent 3
code_implementation_agent = ReactCodeAgent(
    tools=[],
    llm_engine=coding_model_path,
    additional_authorized_imports=["numpy", "pandas", "matplotlib.pyplot", "seaborn", "sklearn"],
)
managed_agent3 = ManagedAgent(
    agent=code_implementation_agent,
    name="code_implementation",
    description="Translate hypothesis into actionable python code. Runs the required analysis and reports back on them."
)

# Agent 4
conclusion_generation_agent = ReactJsonAgent(
    tools=[],
    llm_engine=reasoning_model_path
)
managed_agent4 = ManagedAgent(
    agent=conclusion_generation_agent,
    name="conclusion_generation",
    description="Use the results from the analysis to generate conclusions and export them in markdown"
)

#%%

#=======
# Define a manager
#=======

llm_engine = HfApiEngine(model=reasoning_model_path)

manager_agent = ReactCodeAgent(
    tools=[], 
    llm_engine=llm_engine, 
    managed_agents=[managed_agent1,
                    managed_agent2,
                    managed_agent3,
                    managed_agent4 
                    ]
)

# %%

task = f"Analyze the california_housing.csv file in the path: {data_path}"
manager_agent.run(task)

# %%
