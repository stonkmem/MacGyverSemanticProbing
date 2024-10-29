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

# extract variables from the data into a text file

def export_to_txt(txt_file_path):
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

    def write_nested(data, txt_file, indent=0):
        for key, value in data.items():
            if isinstance(value, dict):
                txt_file.write(f"{' ' * indent}{key}:\n")
                write_nested(value, txt_file, indent + 4)
            elif isinstance(value, list):
                txt_file.write(f"{' ' * indent}{key}:\n")
                for item in value:
                    if isinstance(item, (dict, list)):
                        write_nested({f"item": item}, txt_file, indent + 4)
                    else:
                        txt_file.write(f"{' ' * (indent + 4)}- {item}\n")
            else:
                txt_file.write(f"{' ' * indent}{key}: {value}\n")

    # Write the data to a text file
    with open(txt_file_path, 'w') as txt_file:
        write_nested(outputdict, txt_file)

# Example use case
export_to_txt("outputllama3.1_8B_test.txt")



# def export_to_txt(txt_file_path):
#     outputdict = {   
#         "feasibilitypercentage": check_feasibility(),
#         "efficiencypercentage": check_efficiency(),
#         "SE_simple": SE_simple, 
#         "SE_complex": SE_complex, 
#         "factuality": factuality, 
#         "feasibility": feasibility, 
#         "efficiency": efficiency, 
#         "total_scores": total_scores, 
#         "lambda_scores": lambda_scores,
#         "classprobabilities": classprobabilities, 
#         #    "fullscale_subresponselist": fullscale_subresponselist,
#         "fullscale_classifiedsubresponselist": fullscale_classifiedsubresponselist,
#         "fullscale_classifiedproblist": fullscale_classifiedproblist,
#         "fullscale_promptlist": fullscale_promptlist,
#         "fullscale_prev_steps": fullscale_prev_steps
#     }

#     # Write the data to a text file
#     with open(txt_file_path, 'w') as txt_file:
#         for key, value in outputdict.items():
#             txt_file.write(f"{key}: {value}\n")

# # Export to text file
# export_to_txt("results.txt")

# def export_to_txt(json_file_path, txt_file_path):
#     # Read the JSON file
#     with open(json_file_path, 'r') as json_file:
#         data = json.load(json_file)
    
#     # Write the data to a text file
#     with open(txt_file_path, 'w') as txt_file:
#         for key, value in data.items():
#             txt_file.write(f"{key}: {value}\n")

# # Usage
# export_to_txt("results.json", "results.txt")
# output_file = open("results.json", "w")


# outputdict = {   
#     "feasibilitypercentage": check_feasibility(),
#     "efficiencypercentage": check_efficiency(),
#     "SE_simple": SE_simple, 
#     "SE_complex": SE_complex, 
#     "factuality": factuality, 
#     "feasibility": feasibility, 
#     "efficiency": efficiency, 
#     "total_scores": total_scores, 
#     "lambda_scores": lambda_scores,
#     "classprobabilities": classprobabilities, 
#     #    "fullscale_subresponselist": fullscale_subresponselist,
#     "fullscale_classifiedsubresponselist": fullscale_classifiedsubresponselist,
#     "fullscale_classifiedproblist": fullscale_classifiedproblist,
#     "fullscale_promptlist": fullscale_promptlist,
#     "fullscale_prev_steps": fullscale_prev_steps
# }

# json.dump(outputdict, output_file)
# output_file.close()