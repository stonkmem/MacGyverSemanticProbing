import numpy as np
import sys
# fullscale_classifiedproblist = []
classprobabilities = [] 
# if len(sys.argv) > 1:
#   if sys.argv[1] == 'vicuna' or sys.argv[1] == 'vicuna-7b' or sys.argv[1] == "vicuna-33b":
#     from vicuna_run_benchmark import *
#   elif sys.argv[1] == 'llama' or sys.argv[1] == 'llama_70b' or sys.argv[1] == 'llama3.2' or sys.argv[1] == 'llama30' or sys.argv[1] == 'llama2' or sys.argv[1] == 'llama3.3' or sys.argv[1] == 'llama3-70b' or sys.argv[1] == 'llama3.21b':
#     from Llama_run_benchmark import *
#   elif sys.argv[1] == 'gpt4':
#     from GPT_run_benchmark import *
#   elif sys.argv[1] == 'mixtral' or sys.argv[1] == 'mistral-nemo' or sys.argv[1] == 'mistral-large' or sys.argv[1] == 'ministral' or sys.argv[1] == 'mistral':
#     from Mixtral_run_benchmark import *
#   else:
#     from Llama_run_benchmark import *
# else:
#   from Llama_run_benchmark import *

from src.benchmark import *

print(sys.argv, "SYS ARGV")

use_chateval = False

if len(sys.argv) > 3: 
  if sys.argv[3] == "chateval":
    use_chateval = True
  
print("PROCESSING DATA")
# ??

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
# print(classprobabilities, "classprobabilities")
# print(fullscale_classifiedsubresponselist, "fullscale_classifiedsubresponselist")
# print(fullscale_classifiedproblist, "fullscale_classifiedproblist")

classprobabilitiesN = []

for j in range(len(fullscale_classifiedproblist)): # full scale
  problemscale_classprobabilities = []
  for k in range(len(fullscale_classifiedproblist[j])): # problem scale
    subresponsescale_classprobs = []
    # for each subresponse, there should be an array of class probs.
    for i in range(len(fullscale_classifiedproblist[j][k])): # subresponse scale
      # print(len(fullscale_classifiedproblist[j][k][1])) # should return a nested array containing arrays of probs for each seq in a class.
      classprob = calculate_prob_of_class_logprobsN(fullscale_classifiedproblist[j][k][i])
      # currently configured such that 1 class is 1 problem.
      subresponsescale_classprobs.append(classprob)
    problemscale_classprobabilities.append(subresponsescale_classprobs)

  classprobabilitiesN.append(problemscale_classprobabilities)

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

SE_complexN = []
for i in range(len(classprobabilitiesN)):
  problem_SE = []
  for j in range(len(classprobabilitiesN[i])):
    problem_SE.append(calculate_SE_complex(classprobabilitiesN[i][j]))
  SE_complexN.append(problem_SE)
# SE_complex = SE_complex2 # toggle normalisation 


judge = True
if len(sys.argv) > 5:
  if sys.argv[5] == "false":
    judge = False

if judge:
  # factuality score generation
  factuality = []
  efficiency = []
  feasibility = []
  criterialist = ["feasibility", "safety", "efficiency", "effectiveness"] # add constraint fitting? 
  privector = {
      "feasibility": 0.47295,
      "safety": 0.29784,
      "efficiency": 0.086711,
      "effectiveness": 0.14250
  }

  print(SE_complex, "SE_complex")
  print("LLMJUDGING")
  response_eval_pairs = []
  response_eval_pairs2 = []
  factuality2 = []
  efficiency2 = []
  feasibility2 = []
  for i in range(len(SE_complex)): # for each problem
    solution = " ".join(fullscale_prev_steps[i])
    if use_chateval:
        factual2, feasible2, efficient2, scorearrays2 = gen_factuality_score_chateval_likert(fullscale_promptlist[i][0], solution, criterialist, privector)
    else:
        factual2, feasible2, efficient2, scorearrays2 = gen_factuality_score_likert(fullscale_promptlist[i][0], solution, criterialist)
    factuality2.append(factual2)
    # print(solution, fullscale_promptlist[i][0])
    response_eval_pairs2.append(
          {
              "response": solution,
              "scores": scorearrays2,
              "prompt": fullscale_promptlist[i][0]
          }
      )
    if feasible2 == True:
        feasibility2.append(1)
    else:
        feasibility2.append(0)
    if efficient2 == True:
        efficiency2.append(1)
    else:
        efficiency2.append(0)



    problem_factuality = []
    problem_feasibility = 0
    problem_efficiency = 0
    # aggregates the feasibility and efficiency scores for each problem. 
    factuality.append(problem_factuality)
    if len(SE_complex[i]) > 0:
      if problem_feasibility / len(SE_complex[i]) > 0.6:
        feasibility.append(1)
      else:
        feasibility.append(0)
      if problem_efficiency / len(SE_complex[i]) > 0.6:
        efficiency.append(1)
      else:
        efficiency.append(0)



  total_scores = []
  # for i in range(len(SE_complex)):
  #   problem_scores = []
  #   for j in range(len(SE_complex[i])):
  #     print(compute_total_score(SE_complex[i][j], factuality[i][j]))
  #     problem_scores.append(compute_total_score(SE_complex[i][j], factuality[i][j])) # here factuality is a list of scores while SE is a single value
  #   total_scores.append(problem_scores)
  # # print(total_scores)
  true_total_scores = []
  # for i in range(len(total_scores)):
  #   print(generate_problem_score_simple(total_scores[i]), "generating simple score")
  #   true_total_scores.append(generate_problem_score_simple(total_scores[i]))


  true_total_scores_2 = []
  for i in range(len(SE_complex)):
      print("T2: ", compute_total_score_2(SE_complex[i], factuality2[i]))
      true_total_scores_2.append(compute_total_score_2(SE_complex[i], factuality2[i]))
  print(true_total_scores_2)


  gamma = 0.9  # Discount factor
  lambda_ = 0.8  # Lambda parameter
  lambda_scores = []
  # for i in range(len(total_scores)):
  #   problem_lambda_score = total_lambda_score(total_scores[i], gamma, lambda_)
  #   lambda_scores.append(problem_lambda_score)
  # print(lambda_scores)

  feasibility_score = check_feasibility(feasibility)
  efficiency_score = check_efficiency(efficiency)

  feasibility_score2 = check_feasibility(feasibility2)
  efficiency_score2 = check_efficiency(efficiency2)

print("DATA PROCESSED")
