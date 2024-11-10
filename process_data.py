# calculates the probabilities of each class for each subresponse

import numpy as np
from llama_funcs import *
from helper_funcs import *
from data import *
from openai_funcs import *
from Llama_run_benchmark import * # can mod for other LLMs
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import numpy as np
fullscale_classifiedproblist = []
classprobabilities = [] 
for j in range(len(fullscale_classifiedproblist)): # full scale
  problemscale_classprobabilities = []
  for k in range(len(fullscale_classifiedproblist[j])): # problem scale
    subresponsescale_classprobs = []
    # for each subresponse, there should be an array of class probs.
    for i in range(len(fullscale_classifiedproblist[j][k])): # subresponse scale
      # print(len(fullscale_classifiedproblist[j][k][1])) # should return a nested array containing arrays of probs for each seq in a class.
      classprob = calculate_prob_of_class_logprobs(fullscale_classifiedproblist[j][k][i])
      # currently configured such that 1 class is 1 problem.
      subresponsescale_classprobs.append(classprob)
    problemscale_classprobabilities.append(subresponsescale_classprobs)

  classprobabilities.append(problemscale_classprobabilities)

SE_simple = []
for i in range(len(classprobabilities)):
  problem_SE = []
  for j in range(len(classprobabilities[i])):
    problem_SE.append(calculate_SE_simple(classprobabilities[i][j]))
  print(problem_SE)
  SE_simple.append(problem_SE)


SE_complex = []
for i in range(len(classprobabilities)):
  problem_SE = []
  for j in range(len(classprobabilities[i])):
    problem_SE.append(calculate_SE_complex(classprobabilities[i][j]))
  SE_complex.append(problem_SE)

# factuality score generation
factuality = []
efficiency = []
feasibility = []
criterialist = ["feasibility", "safety", "efficiency", "effectiveness"] # add constraint fitting? 
for i in range(len(SE_complex)): # for each problem
  problem_factuality = []
  problem_feasibility = 0
  problem_efficiency = 0
  for j in range(len(SE_complex[i])): # for each step
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
    if step_feasibility / len(fullscale_subresponselist[i][j]) > 0.6:
        problem_feasibility += 1
    if step_efficiency / len(fullscale_subresponselist[i][j]) > 0.6:
        problem_efficiency += 1
    print(step_feasibility, step_efficiency)

  # aggregates the feasibility and efficiency scores for each problem. 
  factuality.append(problem_factuality)
  if problem_feasibility / len(SE_complex[i]) > 0.6:
    feasibility.append(1)
  else:
    feasibility.append(0)
  if problem_efficiency / len(SE_complex[i]) > 0.6:
    efficiency.append(1)
  else:
    efficiency.append(0)



total_scores = []
for i in range(len(SE_complex)):
  problem_scores = []
  for j in range(len(SE_complex[i])):
    print(compute_total_score(SE_complex[i][j], factuality[i][j]))
    problem_scores.append(compute_total_score(SE_complex[i][j], factuality[i][j])) # here factuality is a list of scores while SE is a single value
  total_scores.append(problem_scores)
# print(total_scores)
for i in range(len(total_scores)):
  print(generate_problem_score_simple(total_scores[i]))
gamma = 0.9  # Discount factor
lambda_ = 0.8  # Lambda parameter
lambda_scores = []
for i in range(len(total_scores)):
  problem_lambda_score = total_lambda_score(total_scores[i], gamma, lambda_)
  lambda_scores.append(problem_lambda_score)
# print(lambda_scores)

feasibility_score = check_feasibility(feasibility)
efficiency_score = check_efficiency(efficiency)