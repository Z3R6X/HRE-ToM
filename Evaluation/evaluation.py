import argparse
import os
import sys
import json
import pandas as pd
from tabulate import tabulate
#sys.path.append(os.path.relpath("ReCEval"))
from util.data import create_directory


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None, type=str)
    parser.add_argument("--evaluator", default="llognet", type=str)
    parser.add_argument("--return_full_scores", action="store_true")
    parser.add_argument("--cache_dir", default="/data/hleier/MA/", type=str)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--apply_splitting", action="store_true")
    args = parser.parse_args()

    # Load the desired evaluation framework
    if args.evaluator == "llognet":
        from eval_frameworks.llognet import LLogNet
        evaluator = LLogNet(
            verbose = args.verbose,
            load_in_8bit = args.load_in_8bit,
            load_in_4bit = args.load_in_4bit,
            apply_splitting = args.apply_splitting,
            return_full_scores = args.return_full_scores
        )

    elif args.evaluator == "roscoe":
        from eval_frameworks.roscoe import Roscoe
        evaluator = Roscoe(
            return_full_scores = args.return_full_scores
        )

    elif args.evaluator == "receval":
        from eval_frameworks.receval import ReCEval
        evaluator = ReCEval(
            return_full_scores = args.return_full_scores
        )

    # Create evaluation sub-directory in the run directory
    working_dir = args.run_dir
    prediction_dir = os.path.join(working_dir, "predictions")
    eval_dir = os.path.join(working_dir, "evaluation_" + evaluator.name)
    create_directory(eval_dir)
    summary_dir = os.path.join(eval_dir, "summaries")
    create_directory(summary_dir)

    # List all JSON files in the prediction sub-directory
    prediction_files = list_prediction_files(prediction_dir)

    summary = {}

    for prediction_file in prediction_files:

        print(f"\nEvaluating \"{prediction_file}\"")

        # Load the predictions to the evaluation framework
        predictions = json.load(open(os.path.join(prediction_dir, prediction_file)))
        evaluator.load_predictions(predictions)

        # Evaluate the predictions
        if args.evaluator == "llognet":
            results = evaluator.evaluate_all()
        else:
            results = evaluator.evaluate_all()
        
        # Save the results of the evaluation
        with open(os.path.join(eval_dir, prediction_file), "w") as outfile:
            outfile.write(json.dumps(results, indent=4))

        # Create summary of results
        summary_task = create_summary(results)
        summary[os.path.splitext(prediction_file)[0]] = summary_task

    # Save evaluation summary as JSON
    with open(os.path.join(summary_dir, "summary_" + evaluator.name + ".json"), "w") as outfile:
        outfile.write(json.dumps(summary, indent=4))

    # Transform summary into table format
    table_dict = create_table_dict(summary)

    # Save evaluation summary as CSV
    df = pd.DataFrame(table_dict)
    df.to_csv(os.path.join(summary_dir, "summary_" + evaluator.name + ".csv"))

    # Save evaluation summary as txt
    table = tabulate(df, headers="keys", tablefmt="grid", showindex=False)
    with open(os.path.join(summary_dir, "summary_" + evaluator.name + ".txt"), "a") as f:
        f.write(f"Evaluation Scores - {working_dir}\n")
        f.write(table)
    
    # Print evaluation summary
    print(table)
    #print(tabulate(df, headers="keys", tablefmt="github"))


def list_prediction_files(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    return json_files


def create_summary(result_dict):

    summary_dict = {}

    #print(result_dict.keys())

    # Extract mean value for each metric
    for key in result_dict.keys():
        summary_dict[key] = result_dict[key]["means"]

    return summary_dict


def create_table_dict(summary_dict):
    table_dict = {}

    for task in summary_dict.keys():

        table_dict.setdefault("task", []).append(task)

        for metric_type in summary_dict[task].keys():

            for metric in summary_dict[task][metric_type].keys():

                table_dict.setdefault(metric, []).append(summary_dict[task][metric_type][metric])

    return table_dict


if __name__ == "__main__":
    main()