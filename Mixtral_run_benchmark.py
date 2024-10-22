

from llama_funcs import *
from helper_funcs import *
from data import *
from openai_funcs import *

# FastLanguageModel.for_inference(model) # Enable native 2x faster inferenc
# need to modify for multi step usage / multi problem usage
fullscale_tokenlist = [] # stores lists of tokens one for each subsequence
fullscale_problist = [] # stores probabilities of tokens, where one element is a list of probabilities for each token in a seq
fullscale_promptlist = [] # list of overall probs for eqch sequence
fullscale_responselist = []
fullscale_subresponselist = [] # is this needed?
fullscale_stepprobs = [] # list of overall probs for each step, for each problem. .
fullscale_prev_steps = []
fullscale_classifiedsubresponselist = []
fullscale_classifiedproblist = []

max_stepnum = 10
min_stepnum = 3

# responses = []
for i in range(1): # handles multiple problems.
  prev_steps = []
  problemscale_problist = []
  problemscale_tokenlist = []
  problemscale_subresponselist = []
  problemscale_stepprobs = []
  problemscale_responselist = []
  problemscale_promptlist = []
  problemscale_classifiedsubresponselist = []
  problemscale_classifiedproblist = []

  max_stepnum = 10
  max_steps = num_to_string[max_stepnum]

  inputstring = f'''
  You are Macgyver, an intelligent person skilled in using ordinary tools in unconventional ways to solve problems.
    Given the problem below, generate a multi-step solution considering all the constraints.
    Solve the problem in the fewest steps possible.


    Be clear, specific and concise, and try to use the items in creative and innovative ways while maintaining practicality.
    Ensure that each step you generate brings you significantly closer to solving the problem fully.
    Do NOT include explanation or examples or code in your response.

    Respond STRICTLY in this format, and do not generate anything extra:
    "Step {1}: <generate step {1} here>"
    "Step {2}: <generate step {2} here>"
    ...

    The complete solution cannot have more than {max_stepnum} steps.
  ''' + extract_problem(macgyver[i]["text"] + "\n ### Response: ")
  print("INPUTSTRING: ", inputstring)

  # generates an initial solution to extract step count.
    
  response, token, prob = gen_prob(inputstring, 1)
  
  while response[0].count('\n') >= 20:
      response, token, prob = gen_prob(inputstring, 1)
      print("REGENERATING")
  response = response[0]
  try:
      response_index = response.index("<|eot_id|>")
      response = response[response_index:]
  except:
    print('INIT:', response)

  steps = split_by_sequence(response, "Step ")
  print("STEPS: ", steps)
  num_steps = len(steps)
  # print("NUM_STEPVERS: ", num_steps)

  num_steps = max(min(max_stepnum, num_steps), min_stepnum)
#   num_steps = min(10, max_stepnum)
  print("MAX_STEPNUM: ", num_steps)

  if num_steps <= 10:
    max_steps = num_to_string[num_steps]
    print("MAX_STEPS: ", max_steps)

  for j in range(num_steps): # handles multiple steps for a problem.
    problem_break = False
    step_num = 1 + j

    if step_num == 1:
      # inputs = tokenizer(
      #   [
      #   macgyver[i]["text"] + "\n ### Response: "
      #   ], return_tensors = "pt").to("cuda")
        print()
    else: # handles further steps
      dictionary = {
          f"Step {2},": f"Step {step_num + 1},",
          f"Step {2 - 1}": f"Step {step_num}",
          f"step {2},": f"step {step_num + 1},",
          f"step {2 - 1}": f"step {step_num}"
      }

      finalstring = replace_all(macgyver[i]["text"], dictionary) # updating prompt

      # updating prompt by appending to prev step list.

      # Currently using greedy decoding
      selected_step_index = max(problemscale_stepprobs[step_num - 2])
      selected_step_index = problemscale_stepprobs[step_num - 2].index(selected_step_index)
      prev_steps.append(f"\n Step {step_num - 1} of the solution is: " + split_by_sequence(problemscale_responselist[step_num - 2], "Step " + str(step_num - 1) + ":")[selected_step_index].replace("Step " + str(step_num - 1) + ":", ""))
      for k in range(len(prev_steps)):
        finalstring += prev_steps[k]
      if step_num >= num_steps:
        finalstring += "\n This step must make the solution complete and solve the problem. "
      finalstring += f"\n ### Response: "
      print("INPUT: ", finalstring)

    stepscale_tokenlist = []
    stepscale_problist = []
    stepscale_subresponselist = []
    stepscale_stepprobs = []

    # gets output from LLM
    
    if step_num == 1:
        inputstring = macgyver[i]["text"] + "\n ### Response: "
    else:
        inputstring = finalstring
    subresponses, tokenlist, problist = gen_prob(inputstring, num_stepvers, verify=True)
    num_stops = 0
    for n in range(len(subresponses)):

      # removing the prompt from the response
      try:
          subresponse_index = subresponses[n].index("<|eot_id|>")
          subresponses[n] = subresponses[n][subresponse_index:]
      except:
        print()
      

      if "STOP" in subresponses[n]:
        num_stops += 1
        print("STOP FOUND")
      else:
        # handle exceptions and different answer formats
        try:
          subresponse_index = subresponses[n].index("Step " + str(step_num) + ":")
          subresponses[n] = subresponses[n][subresponse_index:]
          subresponses[n] = subresponses[n].split('\n')[0] # only first line
          # processing token and prob lists
          
          start_index = tokenlist[n].index('Step')
          tokenlist[n] = tokenlist[n][start_index:]
          problist[n] = problist[n][start_index:]

          try:
              line_index = tokenlist[n].index('\n')
              tokenlist[n] = tokenlist[n][:line_index]
              problist[n] = problist[n][:line_index]
          except:
              print()
          print(tokenlist[n])
          print(problist[n])
        except:
          try:
            subresponse_index = subresponses[n].index(str(step_num) + ":")
            subresponses[n] = "Step " + subresponses[n][subresponse_index:]
          except:
            try:
              subresponse_index = subresponses[n].index("Response: ")
              subresponses[n] = "Step " + str(step_num) + ": " + subresponses[n][subresponse_index:]
            except:
              print("ERROR: ", subresponses[n])
              continue
        subresponses[n] = subresponses[n].replace("<|eot_id|>", "")
        subresponses[n] = subresponses[n].replace("Response:", "")
        try:
          next_step_index = subresponses[n].index("Step " + str(step_num + 1) + ":")
          subresponses[n] = subresponses[n][:next_step_index]
        except:
          print()
          # continue
        if subresponses[n].count("Step " + str(step_num) + ":") > 1:
            subresponses[n] = remove_duplicates(subresponses[n], "Step " + str(step_num) + ":")

        stepscale_subresponselist.append(subresponses[n])

        # calculating probability of sequence

        overall_probability = calc_seq_probability_LOGPROB(problist[n])
        stepscale_stepprobs.append(overall_probability)
        print(f"Overall Probability for step {step_num}: {overall_probability}")

        # appending to step scale
        stepscale_tokenlist.append(tokenlist[n])
        stepscale_problist.append(problist[n])

      print("SUBRESPONSE: ", subresponses[n])

    print("NUM_STOPS: ", num_stops)
    if num_stops >= num_stepvers / 2:
      problem_break = True
      break

    problemscale_tokenlist.append(stepscale_tokenlist)
    problemscale_problist.append(stepscale_problist)
    problemscale_subresponselist.append(stepscale_subresponselist)
    problemscale_stepprobs.append(stepscale_stepprobs)

    # classifying responses for SE

    prompt = macgyver[i]["text"]
    if step_num != 1:
      prompt = finalstring

    problem_index = prompt.index("Problem:")
    prompt = prompt[:problem_index]
    print("PROMPT: ", prompt)
    # shift classification outside of the main generation loop? 
    classified_response, classified_token, classified_prob = gen_C(prompt, stepscale_subresponselist, stepscale_tokenlist, stepscale_problist)
    print("CLASSIFIED RESPONSE: ")
    for m in range(len(classified_response)):
      print(classified_response[m])
    problemscale_classifiedsubresponselist.append(classified_response)
    problemscale_classifiedproblist.append(classified_prob)
    problemscale_promptlist.append(prompt)

    # constructs the full response
    response = ""
    for m in range(len(stepscale_subresponselist)):
      response += stepscale_subresponselist[m]
    print("INDEXED RESPONSE: ", response)
    # appends the response to the list for the particular problem.
    problemscale_responselist.append(response)

  # appends problem scale lists to full scale lists
#   fullscale_tokenlist.append(problemscale_tokenlist) # not needed
  fullscale_problist.append(problemscale_problist)
#   fullscale_responselist.append(problemscale_responselist) # each element is prompt at each step, this is needed? idts
  fullscale_subresponselist.append(problemscale_subresponselist)
#   fullscale_stepprobs.append(problemscale_stepprobs) # idt needed


  selected_step_index = max(problemscale_stepprobs[step_num - 1])
  selected_step_index = problemscale_stepprobs[step_num - 1].index(selected_step_index)
  prev_steps.append(f"\n Step {step_num} of the solution is: " + split_by_sequence(problemscale_responselist[step_num - 1], "Step " + str(step_num) + ":")[selected_step_index].replace("Step " + str(step_num) + ":", ""))
  
  fullscale_prev_steps.append(prev_steps) # for each problem
  fullscale_promptlist.append(problemscale_promptlist) # needed 
  fullscale_classifiedsubresponselist.append(problemscale_classifiedsubresponselist) # needed
  fullscale_classifiedproblist.append(problemscale_classifiedproblist) # needed 


 # tokenlist is quad nested:
 # [[problem 1], [[step 1]], [[[sub response 1]]], [[[token 1 in subresponse 1 of step 1 of problem 3]]]]




# print(check_feasibility(), check_efficiency())

