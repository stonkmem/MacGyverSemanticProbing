import numpy as np
import sys
from src.benchmark import *

def calculate_class_probabilities(classified_prob_list, use_N=False):
    """
    Calculate class probabilities for each subresponse.
    
    Args:
        classified_prob_list: List of classified probability distributions
        use_N: Boolean indicating whether to use normalized calculation
        
    Returns:
        List of calculated class probabilities
    """
    if use_N:
        calculate_func = calculate_prob_of_class_logprobsN
    else:
        calculate_func = calculate_prob_of_class_logprobs
    
    class_probabilities = []
    for problem_scale in classified_prob_list:
        problem_probs = []
        for subresponse in problem_scale:
            subresponse_probs = []
            for sequence in subresponse:
                class_prob = calculate_func(sequence)
                subresponse_probs.append(class_prob)
            problem_probs.append(subresponse_probs)
        class_probabilities.append(problem_probs)
    return class_probabilities

def calculate_se_values(class_probabilities, calculation_method):
    """
    Calculate entropy values using specified method.
    
    Args:
        class_probabilities: List of class probabilities
        calculation_method: Function to use for SE calculation
        
    Returns:
        List of calculated SE values
    """
    se_values = []
    for problem_probs in class_probabilities:
        problem_se = []
        for subresponse_probs in problem_probs:
            se = calculation_method(subresponse_probs)
            problem_se.append(se)
        se_values.append(problem_se)
    return se_values

def generate_factuality_scores(prompt_list, prev_steps, use_chateval):
    """
    Generate factuality scores using specified evaluation method.
    
    Args:
        prompt_list: List of prompts
        prev_steps: Previous steps in solution process
        use_chateval: Boolean indicating whether to use chat eval
        
    Returns:
        Tuple containing:
            - Response evaluation pairs
            - Factuality scores
            - Feasibility scores
            - Efficiency scores
    """
    response_eval_pairs = []
    factuality_scores = []
    feasibility_scores = []
    efficiency_scores = []
    
    for idx, (prompt, steps) in enumerate(zip(prompt_list, prev_steps)):
        solution = " ".join(steps)
        
        if use_chateval:
            factual, feasible, efficient, scores = gen_factuality_score_chateval_likert(
                prompt, solution, ["feasibility", "safety", "efficiency", "effectiveness"], 
                {"feasibility": 0.47295, "safety": 0.29784, "efficiency": 0.086711, "effectiveness": 0.14250}
            )
        else:
            factual, feasible, efficient, scores = gen_factuality_score_likert(
                prompt, solution, ["feasibility", "safety", "efficiency", "effectiveness"]
            )
        
        response_eval_pairs.append({
            "response": solution,
            "scores": scores,
            "prompt": prompt
        })
        
        factuality_scores.append(factual)
        feasibility_scores.append(1 if feasible else 0)
        efficiency_scores.append(1 if efficient else 0)
    
    return response_eval_pairs, factuality_scores, feasibility_scores, efficiency_scores

def process_data():
    """
    Main data processing function.
    """
    print(sys.argv, "SYS ARGV")
    use_chateval = len(sys.argv) > 3 and sys.argv[3] == "chateval"
    
    # Calculate class probabilities
    class_probabilities = calculate_class_probabilities(fullscale_classifiedproblist)
    class_probabilitiesN = calculate_class_probabilities(fullscale_classifiedproblist, use_N=True)
    
    # Calculate SE values
    se_simple = calculate_se_values(class_probabilities, calculate_SE_simple)
    se_complex = calculate_se_values(class_probabilities, calculate_SE_complex)
    se_complexN = calculate_se_values(class_probabilitiesN, calculate_SE_complex)
    
    # Generate factuality scores if enabled
    judge_enabled = len(sys.argv) > 5 and sys.argv[5] != "false"
    if judge_enabled:
        response_eval_pairs, factuality_scores, feasibility_scores, efficiency_scores = generate_factuality_scores(
            fullscale_promptlist, fullscale_prev_steps, use_chateval
        )
        
        # Calculate total scores
        true_total_scores_2 = [compute_total_score_2(se_complex[i], factuality_scores[i]) 
                              for i in range(len(se_complex))]
        print(true_total_scores_2)
        
        feasibility_score = check_feasibility(feasibility_scores)
        efficiency_score = check_efficiency(efficiency_scores)
    
    print("DATA PROCESSED")

if __name__ == "__main__":
    process_data()