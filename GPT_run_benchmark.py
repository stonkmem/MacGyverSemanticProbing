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
min_steps = 3

# responses = []
for i in range(5): # handles multiple problems.
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
    Do not include explanation in your response.

    Respond STRICTLY in this format, and do not generate anything extra:
    "Step {1}: <generate step {1} here>"
    "Step {2}: <generate step {2} here>"
    ...

    The complete solution cannot have more than {max_steps} steps.
  ''' + extract_problem(macgyver[i]["text"] + "\n ### Response: ")
  print("INPUTSTRING: ", inputstring)

  inputs = tokenizer(
    [
    inputstring
    ]
    , return_tensors = "pt").to("cuda")

  # generates an initial solution to extract step count.

  response, token, prob = generate_data_from_GPT(1, inputstring) # generate_tokens_and_probabilities(inputs)
  response = response[0]
  token = token[0]
  prob = prob[0]
  try:
      response_index = response.index("<|eot_id|>")
      response = response[response_index:]
  except:
      print(response)

  steps = split_by_sequence(response, "Step ")
  print("STEPS: ", steps)
  num_steps = len(steps)
  # print("NUM_STEPVERS: ", num_steps)

  max_stepnum = max(min(max_stepnum, num_steps), min_steps)
  max_stepnum = min(10, max_stepnum)
  print("MAX_STEPNUM: ", max_stepnum)

  if num_steps <= 10:
    max_steps = num_to_string[num_steps]
    print("MAX_STEPS: ", max_steps)

  for j in range(max_stepnum): # handles multiple steps for a problem.
    problem_break = False
    step_num = 1 + j

    if step_num == 1:
      inputs = tokenizer(
        [
        macgyver[i]["text"] + "\n ### Response: "
        ], return_tensors = "pt").to("cuda")
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
      # print("SELECTED STEP INDEX: ", selected_step_index, problemscale_stepprobs[step_num - 2])
      # print(split_by_sequence(problemscale_responselist[step_num - 2], "Step " + str(step_num - 1) + ":"))
      prev_steps.append(f"\n Step {step_num - 1} of the solution is: " + split_by_sequence(problemscale_responselist[step_num - 2], "Step " + str(step_num - 1) + ":")[selected_step_index].replace("Step " + str(step_num - 1) + ":", ""))
      for k in range(len(prev_steps)):
        finalstring += prev_steps[k]
      if step_num >= max_stepnum:
        finalstring += "\n This step must make the solution complete and solve the problem. "

      finalstring += f"\n ### Response: "
      print("INPUT: ", finalstring)
      inputs = tokenizer(
        [
            finalstring
        ]
        , return_tensors = "pt").to("cuda")

    stepscale_tokenlist = []
    stepscale_problist = []
    stepscale_subresponselist = []
    stepscale_stepprobs = []

    # gets output from LLM
    if step_num == 1:
        finalstring = macgyver[i]["text"] + "\n ### Response: "
    subresponses, tokenlist, problist = generate_data_from_GPT(num_stepvers, finalstring)
    num_stops = 0
    for n in range(len(subresponses)):

      # removing the prompt from the response
      try:
          subresponse_index = subresponses[n].index("<|eot_id|>")
          subresponses[n] = subresponses[n][subresponse_index:]
      except:
          print("GPT_RESPONSE")

      if "STOP" in subresponses[n]:
        num_stops += 1
        print("STOP FOUND")
      else:
        # handle exceptions and different answer formats
        try:
          subresponse_index = subresponses[n].index("Step " + str(step_num) + ":")
          subresponses[n] = subresponses[n][subresponse_index:]
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
        except:
          print()
        if subresponses[n].count("Step " + str(step_num) + ":") > 1:
            subresponses[n] = remove_duplicates(subresponses[n], "Step " + str(step_num) + ":")

        stepscale_subresponselist.append(subresponses[n])

        # calculating probability of sequence

        overall_probability = calc_sequence_probability_LOGPROB(problist[n])
        stepscale_stepprobs.append(overall_probability)
        print(f"Overall Probability for step {step_num}: {overall_probability}")

        # appending to step scale, a bit redundant.
        stepscale_tokenlist.append(tokenlist[n])
        stepscale_problist.append(problist[n])

      print("SUBRESPONSE: ", subresponses[n])
      print()

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
  # print("SELECTED STEP INDEX: ", selected_step_index, problemscale_stepprobs[step_num - 2])
  # print(split_by_sequence(problemscale_responselist[step_num - 2], "Step " + str(step_num - 1) + ":"))
  prev_steps.append(f"\n Step {step_num} of the solution is: " + split_by_sequence(problemscale_responselist[step_num - 1], "Step " + str(step_num) + ":")[selected_step_index].replace("Step " + str(step_num) + ":", ""))
  
  fullscale_prev_steps.append(prev_steps) # for each problem
  fullscale_promptlist.append(problemscale_promptlist) # needed 
  fullscale_classifiedsubresponselist.append(problemscale_classifiedsubresponselist) # needed
  fullscale_classifiedproblist.append(problemscale_classifiedproblist) # needed 


 # tokenlist is quad nested:
 # [[problem 1], [[step 1]], [[[sub response 1]]], [[[token 1 in subresponse 1 of step 1 of problem 3]]]]
print(problemscale_responselist)
print(problemscale_tokenlist)
print(problemscale_problist)
print(problemscale_subresponselist)
print(problemscale_stepprobs)
print(problemscale_classifiedsubresponselist)
print(fullscale_prev_steps)

# gets class probabilities

classprobabilities = [] # currently for one problem.
# print(fullscale_classifiedproblist)
for j in range(len(fullscale_classifiedproblist)): # full scale
  problemscale_classprobabilities = []
  for k in range(len(fullscale_classifiedproblist[j])): # problem scale
    subresponsescale_classprobs = []
    # for each subresponse, there should be an array of class probs.
    for i in range(len(fullscale_classifiedproblist[j][k])): # subresponse scale
      # print(len(fullscale_classifiedproblist[j][k][1])) # should return a nested array containing arrays of probs for each seq in a class.
      classprob = calculate_prob_of_class_logprobs(fullscale_classifiedproblist[j][k][i], False)
      # print(classprob)
      # currently configured such that 1 class is 1 problem.
      subresponsescale_classprobs.append(classprob)
    problemscale_classprobabilities.append(subresponsescale_classprobs)

  classprobabilities.append(problemscale_classprobabilities)
print(classprobabilities)

# calculates SE

SE = []
for i in range(len(classprobabilities)):
  problem_SE = []
  for j in range(len(classprobabilities[i])):
    problem_SE.append(calculate_SE_simple(classprobabilities[i][j]))
  print(problem_SE)
  SE.append(problem_SE)
print(SE)

for i in range(len(classprobabilities)):
  problem_SE = []
  for j in range(len(classprobabilities[i])):
    problem_SE.append(calculate_SE_complex(classprobabilities[i][j]))
  SE.append(problem_SE)
print(SE)

# factuality score generation
factuality = []
efficiency = []
feasibility = []
criterialist = ["feasible", "safe", "resource-efficient", "effective"] # add constraint fitting? 
for i in range(len(SE)): # for each problem
  problem_factuality = []
  problem_feasibility = 0
  problem_efficiency = 0
  for j in range(len(SE[i])): # for each step
    step_factuality = []
    step_feasibility = 0
    step_efficiency = 0
    for k in range(len(fullscale_subresponselist[i][j])): # for each sub response
        factual, feasible, efficient = gen_factuality_score_likert(fullscale_promptlist[i][j], fullscale_subresponselist[i][j][k], criterialist)
        step_factuality.append(factual)
        if feasible == True:
            step_feasibility += 1
        if efficient == True:
            step_efficiency += 1
    problem_factuality.append(step_factuality)
    if step_feasibility / len(fullscale_subresponselist[i][j]) > 0.5:
        problem_feasibility += 1
    if step_efficiency / len(fullscale_subresponselist[i][j]) > 0.5:
        problem_efficiency += 1
    print(step_feasibility, step_efficiency)

  # aggregates the feasibility and efficiency scores for each problem. 
  factuality.append(problem_factuality)
  if problem_feasibility / len(SE[i]) > 0.6:
    feasibility.append(1)
  else:
    feasibility.append(0)
  if problem_efficiency / len(SE[i]) > 0.6:
    efficiency.append(1)
  else:
    efficiency.append(0)
    
print(factuality)
print(feasibility)
print(efficiency)

total_scores = []
for i in range(len(SE)):
  problem_scores = []
  for j in range(len(SE[i])):
    print(compute_total_score(SE[i][j], factuality[i][j]))
    problem_scores.append(compute_total_score(SE[i][j], factuality[i][j])) # here factuality is a list of scores while SE is a single value
  total_scores.append(problem_scores)
print(total_scores)
for i in range(len(total_scores)):
  print(generate_problem_score_simple(total_scores[i]))
# print(generate_problem_score_simple(total_scores[0]))

