from src.process_data import *
# import feasibility_score, efficiency_score, SE_simple, SE_complex, factuality, feasibility, efficiency, total_scores, lambda_scores, classprobabilities, fullscale_classifiedsubresponselist, fullscale_classifiedproblist, fullscale_promptlist, fullscale_prev_steps, true_total_scores
# feasibility_score, efficiency_score, SE_simple, SE_complex, factuality, feasibility, efficiency, total_scores, lambda_scores, classprobabilities, fullscale_classifiedsubresponselist, fullscale_classifiedproblist, fullscale_promptlist, fullscale_prev_steps

import json
import sys

# extract variables from the data into a JSON file
print("EXPORTING DATA")
output_filename = "results_test1.json"
if len(sys.argv) > 2:
    output_filename = sys.argv[2]
output_file = open(output_filename, "w")

if judge:
    total_scores = []
    outputdict = {  
        "SE_simple": SE_simple, 
        "SE_complex": SE_complex, 
        "SE_complexN": SE_complexN,
        "classprobabilities": classprobabilities, 
        
        "feasibilitypercentage": feasibility_score,
        "efficiencypercentage": efficiency_score,
        "factuality": factuality, 
        "feasibility": feasibility, 
        "efficiency": efficiency, 
        "total_scores": total_scores, 
        "lambda_scores": lambda_scores,
        
        #    "fullscale_subresponselist": fullscale_subresponselist,
        "fullscale_classifiedsubresponselist": fullscale_classifiedsubresponselist,
        "fullscale_classifiedproblist": fullscale_classifiedproblist,
        "fullscale_promptlist": fullscale_promptlist,
        "fullscale_prev_steps": fullscale_prev_steps,
        "fullscale_classifiedstepproblist" : fullscale_classifiedstepproblist,
        "response_eval_pairs" : response_eval_pairs,
        
        "response_eval_pairs_2" : response_eval_pairs2,
        "true_total_scores_2" : true_total_scores_2,
        "factuality2" : factuality2,
        "efficiency2" : efficiency2,
        "feasibility2" : feasibility2,
        "feasibilitypercentage2": feasibility_score2,
        "efficiencypercentage2" : efficiency_score2,

        # "fullscale_hslist": fullscale_hslist,
    }
else:
    toggle_hs = False
    if len(sys.argv) > 8:
        if sys.argv[8] == 'hs':
            toggle_hs = True
    if toggle_hs:
        outputdict = { 
            "SE_simple": SE_simple, 
            "SE_complex": SE_complex, 
            "SE_complexN": SE_complexN,
            "classprobabilities": classprobabilities, 
            
            #    "fullscale_subresponselist": fullscale_subresponselist,
            "fullscale_classifiedsubresponselist": fullscale_classifiedsubresponselist,
            "fullscale_classifiedproblist": fullscale_classifiedproblist,
            "fullscale_promptlist": fullscale_promptlist,
            "fullscale_prev_steps": fullscale_prev_steps,
            "fullscale_hslist": fullscale_hslist,
        }
    else:
        outputdict = { 
            "SE_simple": SE_simple, 
            "SE_complex": SE_complex, 
            "SE_complexN": SE_complexN,
            "classprobabilities": classprobabilities, 
            
            #    "fullscale_subresponselist": fullscale_subresponselist,
            "fullscale_classifiedsubresponselist": fullscale_classifiedsubresponselist,
            "fullscale_classifiedproblist": fullscale_classifiedproblist,
            "fullscale_promptlist": fullscale_promptlist,
            "fullscale_prev_steps": fullscale_prev_steps,
        }

json.dump(outputdict, output_file)
output_file.close()