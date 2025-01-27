from src.llama_funcs import *

LLMGenDict = {
  "llama": gen_prob,
  "llama_70b": gen_prob,
  "llama3.2": gen_prob,
  "llama30": gen_prob,
  "llama2": gen_prob,
  "llama3.3": gen_prob,
  "llama3-70b": gen_prob,
  "llama3.21b": gen_prob,
  "vicuna": gen_prob_vicuna,
  "vicuna-7b": gen_prob_vicuna,
  "vicuna-33b": gen_prob_vicuna,
  "mixtral": gen_prob_mistral,
  "mistral-nemo": gen_prob_mistral,
  "mistral-large": gen_prob_mistral,
  "ministral": gen_prob_mistral,
  "mistral": gen_prob_mistral,

}
if len(sys.argv) < 2:
    print("Please specify the LLM to use.")
    sys.exit(0)
LLMGenFunction = LLMGenDict[sys.argv[1]]

step_num = 1
max_stepnum = 10
min_stepnum = 2
# max_steps = 'ten'
prompt = f"""Please act as Macgyver, an intelligent person skilled in using ordinary tools in unconventional ways to solve problems.
    Given the problem below, create ONE possible next step {step_num} to a multi-stage solution considering all the constraints and previous steps, if any.
    Solve the problem in the fewest steps possible.
    Arrive at the complete solution by step {max_steps}, such that it can solve the problem.
    Be clear, specific and concise, and make your step safe, feasible, efficient and effective.
    Ensure that the step you generate brings you significantly closer to solving the problem fully.
    
    Do not generate step {step_num + 1}, etc., or include explanation, examples or any python (or other) code in your response.
    Limit the length of the one step you generate to one sentence maximum.  
    Make your response as creative and innovative as possible.
    
    Respond strictly with "STOP" if you think that the existing steps provided already form a complete solution to the problem. 
    Else, respond STRICTLY in this format:
    Step {step_num}: <generate version of step {step_num} here>

    """
EOS_TOKEN = tokenizer.eos_token

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
fullscale_classifiedstepproblist = []

fullscale_hslist = []



starting_problem = 0
if len(sys.argv) > 9:
  starting_problem = int(sys.argv[9])
num_problems = 1
if len(sys.argv) > 7:
  num_problems = int(sys.argv[7])

toggle_hs = False
if len(sys.argv) > 8:
    if sys.argv[8] == 'hs':
        toggle_hs = True

for a in range(num_problems): # handles multiple problems.
  prev_steps = []
  problemscale_problist = []
  problemscale_tokenlist = []
  problemscale_subresponselist = []
  problemscale_stepprobs = []
  problemscale_responselist = []
  problemscale_promptlist = []
  problemscale_classifiedsubresponselist = []
  problemscale_classifiedproblist = []
  problemscale_classifiedstepprobs = []

  problemscale_hslist = []
  problemscale_finalhslist = []
  i = starting_problem + a

  max_stepnum = 10
  max_steps = num_to_string[max_stepnum]

  inputstring = f'''
  You are Macgyver, an intelligent person skilled in using ordinary tools in unconventional ways to solve problems.
    Given the problem below, generate a multi-step solution considering all the constraints.
    Solve the problem in the fewest steps possible.

    Be clear, specific and concise, and try to use the items in creative and innovative ways while maintaining practicality.
    Ensure that each step you generate brings you significantly closer to solving the problem fully.

    Respond STRICTLY in this format, and do not generate anything extra:
    "Step {1}: <generate step {1} here>"
    "Step {2}: <generate step {2} here>"
    ...

    The complete solution cannot have more than {max_stepnum} steps.
    Do NOT include explanation or examples or code in your response.
  ''' 
#   print("INPUTSTRING: ", inputstring)

  # generates an initial solution to extract step count.
    
  response, token, prob, hs = LLMGenFunction(extract_problem(macgyver[i]["text"] + "\n ### Response: "), inputstring, include_eg = False)
  
  while response[0].count('\n') >= 20 or response.count("Step") >= 15:
      response, token, prob, hs = LLMGenFunction(extract_problem(macgyver[i]["text"] + "\n ### Response: "), inputstring, include_eg = False)
      print("REGENERATING")
  response = response[0]
  # try:
  #     response_index = response.index("<|eot_id|>")
  #     response = response[response_index:]
  # except:
  #   print('INIT:', response)

  steps = split_by_sequence(response, "Step")
  num_steps = response.count("Step")
  num_steps = max(min(max_stepnum, num_steps), min_stepnum)
  if num_steps <= 10:
    max_steps = num_to_string[num_steps]
    print("MAX_STEPS: ", max_steps)
  problem_break = False
  for j in range(num_steps): # handles multiple steps for a problem.
    step_num = 1 + j
    promptstring = prompt
    if step_num != 1: # handles further steps
      dictionary = {
          f"Step {2},": f"Step {step_num + 1},",
          f"Step {2 - 1}": f"Step {step_num}",
          f"step {2},": f"step {step_num + 1},",
          f"step {2 - 1}": f"step {step_num}"
      }

      problemstring = ''
      
      promptstring = replace_all(promptstring, dictionary) # updating prompt
      promptstring = replace_all(
        promptstring, {f"step 101": "step 11", "step 90": "step 10"}
      )

      # updating prompt by appending to prev step list.

      # Currently using greedy decoding
      selected_step_index = max(problemscale_stepprobs[step_num - 2])
      selected_step_index = problemscale_stepprobs[step_num - 2].index(selected_step_index)
      prev_steps.append(f"Step {step_num - 1} of the solution is: " + problemscale_subresponselist[step_num - 2][selected_step_index].replace("Step " + str(step_num - 1) + ":", "") + '\n')
      problemstring = macgyver[i]['Problem'] + '\n' + "Existing steps, if any:\n "
      for k in range(len(prev_steps)):
        problemstring += prev_steps[k]

      problemstring += EOS_TOKEN
      if step_num >= num_steps:
        problemstring += "\n This step must make the solution complete and solve the problem. "
      problemstring += f"\n### Response: "
      if toggle_hs:
        problemscale_finalhslist.append(problemscale_hslist[step_num - 2][selected_step_index])

    stepscale_tokenlist = []
    stepscale_problist = []
    stepscale_subresponselist = []
    stepscale_stepprobs = []

    # gets output from LLM
    
    if step_num == 1:
        problemstring = macgyver[i]["Problem"] + "\n Existing steps, if any:\n " + EOS_TOKEN + "\n### Response: "
#     problemstring += EOS_TOKEN
    subresponses, tokenlist, problist, hs = LLMGenFunction(problemstring, promptstring, num_stepvers, include_eg=False, verify=True)
    num_stops = 0
    for n in range(len(subresponses)):
      # removing the prompt from the response
      try:
          subresponse_index = subresponses[n].index("<|eot_id|>")
          subresponses[n] = subresponses[n][subresponse_index:]
      except Exception:
          pass
      
      if "STOP" in subresponses[n] or ("stop" in subresponses[n].lower() and len(subresponses[n]) < 10):
        num_stops += 1
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
          except Exception:
              pass
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
              
        subresponses[n] = subresponses[n].replace("<|eot_id|>", "")
        subresponses[n] = subresponses[n].replace("Response:", "")
        try:
          next_step_index = subresponses[n].index("Step " + str(step_num + 1) + ":")
          subresponses[n] = subresponses[n][:next_step_index]
        except Exception:
          pass
        if subresponses[n].count("Step " + str(step_num) + ":") > 1:
            subresponses[n] = remove_duplicates(subresponses[n], "Step " + str(step_num) + ":")

        stepscale_subresponselist.append(subresponses[n])

        # calculating probability of sequence

        overall_probability = calc_sequence_probability_LOGPROB(problist[n])
        stepscale_stepprobs.append(overall_probability)
        # print(f"Overall Probability for step {step_num}: {overall_probability}")

        # appending to step scale
        stepscale_tokenlist.append(tokenlist[n])
        stepscale_problist.append(problist[n])

      # print("SUBRESPONSE: ", subresponses[n])

    print("NUM_STOPS: ", num_stops, num_stepvers / 2)
    if num_stops >= num_stepvers / 2:
      # print("STOPPING, NO MORE STEPS")
      problem_break = True
      break

    problemscale_tokenlist.append(stepscale_tokenlist)
    problemscale_problist.append(stepscale_problist)
    problemscale_subresponselist.append(stepscale_subresponselist)
    problemscale_stepprobs.append(stepscale_stepprobs)

    if toggle_hs:
      problemscale_hslist.append(hs)  

    # classifying responses for SE

    prompter = promptstring + 'Problem:\n' + problemstring
    classified_response, classified_token, classified_prob = gen_C(prompter, stepscale_subresponselist, stepscale_tokenlist, stepscale_problist)
    stepscale_classifiedstepprob = []
    for m in range(len(classified_response)):
      print(classified_response[m])
      class_stepprob = []
      for z in range(len(classified_prob[m])):
        overall_prob = calc_sequence_probability_LOGPROB(classified_prob[m][z])
        class_stepprob.append(overall_prob)
      stepscale_classifiedstepprob.append(class_stepprob)
    problemscale_classifiedstepprobs.append(stepscale_classifiedstepprob)

    problemscale_classifiedsubresponselist.append(classified_response)
    problemscale_classifiedproblist.append(classified_prob)
    problemscale_promptlist.append(prompter)

    # constructs the full response
    response = ""
    for m in range(len(stepscale_subresponselist)):
      response += stepscale_subresponselist[m] + '\n'
    # appends the response to the list for the particular problem.
    problemscale_responselist.append(response)
  fullscale_problist.append(problemscale_problist)
  fullscale_subresponselist.append(problemscale_subresponselist)
  fullscale_classifiedstepproblist.append(problemscale_classifiedstepprobs)
  if num_stops < num_stepvers and problem_break == False:
      if len(problemscale_stepprobs[step_num - 1]) >= 1:
          selected_step_index = max(problemscale_stepprobs[step_num - 1])
          selected_step_index = problemscale_stepprobs[step_num - 1].index(selected_step_index)
          prev_steps.append(f"\n Step {step_num} of the solution is: " + problemscale_subresponselist[step_num - 1][selected_step_index].replace("Step " + str(step_num) + ":", ""))
      if toggle_hs:
          problemscale_finalhslist.append(problemscale_hslist[step_num - 1][selected_step_index])

  fullscale_hslist.append(problemscale_finalhslist)
  fullscale_prev_steps.append(prev_steps) # for each problem
  fullscale_promptlist.append(problemscale_promptlist) # needed 
  fullscale_classifiedsubresponselist.append(problemscale_classifiedsubresponselist) # needed
  fullscale_classifiedproblist.append(problemscale_classifiedproblist) # needed 

  now = datetime.now()

  current_time = now.strftime("%H:%M:%S")
  print("Current Time =", current_time, "problem end")

