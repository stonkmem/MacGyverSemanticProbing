import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('https://github.com/allenai/MacGyver/raw/refs/heads/main/data/MacGyver/problem_solution_pair.xlsx')
df_extra = pd.read_excel('https://github.com/allenai/MacGyver/raw/refs/heads/main/data/MacGyver/additional_human_solutions.xlsx')
df.to_csv('MacGyver.csv')
df_extra.info()


# %%
# import pandas as pd
# import numpy as np
# import seaborn as sns
df = pd.read_excel("https://github.com/allenai/MacGyver/blob/main/data/MacGyver/problem_solution_pair.xlsx?raw=true", engine="openpyxl")
df.head()
df.to_csv('problem_solution_pair.csv')

macgyver_prompt = """
{}

### Problem:
{}

### Existing steps, if any:
{}"""


num_stepvers = 11
step_num = 1
max_steps = "ten"

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def format_macgyver_prompt(examples): # Note that {n} steps have already been given beneath the problem, and are required to generate the new step when those steps have been completed. Change first into {n}?
    # and try to use the items in creative and innovative ways while
    instructions = f"""Please act as Macgyver, an intelligent person skilled in using ordinary tools in unconventional ways to solve problems.
    Given the problem below, create ONE possible next step {step_num} to a multi-stage solution considering all the constraints and previous steps, if any.
    Solve the problem in the fewest steps possible.
    Arrive at the complete solution by step {max_steps}, such that it can solve the problem.
    Be clear, specific and concise, maintaining practicality.
    Ensure that the step you generate brings you significantly closer to solving the problem fully.
    Do not include explanation in your response.
    Do not generate step {step_num + 1}, etc.
    Do NOT generate anything extra other than the one step, and limit the length of the one step you generate to one sentence maximum.
    Make your response creative and innovative.

    Respond STRICTLY in this format:
    Step {step_num}: <generate version of step {step_num} here>

    If a new step does not need to be generated to solve the problem, respond strictly with "STOP"
    """ # is the solution complete? if yes, reply only with "Complete: " and "Y", and reply only with "Complete: N" for no.
    # examples["instruction"]
    inputs = examples["Problem"]
    solvable = examples["Solvable?"]
    unconventional = examples["Unconventional?"]
    outputs = examples["Solution"]
    texts = []
    for input, solvable, unconventional, output in zip(inputs, solvable, unconventional, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        # format is an inbuilt py funtion
        text = macgyver_prompt.format(instructions, input, "") + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass




# %%
from datasets import load_dataset
macgyver = load_dataset("csv", data_files="problem_solution_pair.csv", split="train")
macgyver = macgyver.map(format_macgyver_prompt, batched = True,)
macgyver = macgyver.filter(lambda example: example["Solvable?"] == "Yes")
print(macgyver[0]["text"])