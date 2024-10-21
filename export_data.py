# extract variables from the data into a JSON file
# extract variables from the data into a JSON file

from llama_funcs import *
from helper_funcs import *
from data import *
from openai_funcs import *
from Llama_run_benchmark import *
from openai_funcs import calculate_prob_of_class_logprobs
from process_data import *

import json
output_file = open("results.json", "w")


outputdict = {   
    "feasibilitypercentage": check_feasibility(),
    "efficiencypercentage": check_efficiency(),
    "SE_simple": SE_simple, 
    "SE_complex": SE_complex, 
    "factuality": factuality, 
    "feasibility": feasibility, 
    "efficiency": efficiency, 
    "total_scores": total_scores, 
    "lambda_scores": lambda_scores,
    "classprobabilities": classprobabilities, 
    #    "fullscale_subresponselist": fullscale_subresponselist,
    "fullscale_classifiedsubresponselist": fullscale_classifiedsubresponselist,
    "fullscale_classifiedproblist": fullscale_classifiedproblist,
    "fullscale_promptlist": fullscale_promptlist,
    "fullscale_prev_steps": fullscale_prev_steps
}

json.dump(outputdict, output_file)
output_file.close()