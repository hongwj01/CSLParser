"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import json
import os
import re
import time
import csv
from multiprocessing import Process

from tqdm import tqdm
from evaluation.utils.common import correct_templates_and_update_files
from evaluation.utils.GA_calculator import evaluate
from evaluation.utils.template_level_analysis import evaluate_template_level, evaluate_template_level_lstm
from evaluation.utils.PA_calculator import calculate_parsing_accuracy, calculate_parsing_accuracy_lstm
import pandas as pd
from .post_process import correct_single_template

# TIMEOUT = 3600 * 12  # log template identification timeout (sec)
TIMEOUT = 3600 * 48  # log template identification timeout (sec)


def prepare_results(output_dir):
    if not os.path.exists(output_dir):
        # make output directory
        os.makedirs(output_dir)

    # make a new summary file
    result_file = 'parsing_accuracy.csv'
    if not os.path.exists(os.path.join(output_dir, result_file)):
        with open(os.path.join(output_dir, result_file), 'w') as csv_file:
            fw = csv.writer(csv_file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            fw.writerow(['Dataset', 'TrainingTime', 'ParsingTime', 'IdentifiedTemplates', 'GroundTemplates', 'GA', 'FGA', 'PA', 'FTA', 'PTA', 'RTA', 'CacheTime', 'PreTrainedModelTime', 'LLM_Time', 'LLM_Merge_Time', 'LLM_Total_Time', 'LLM_Tokens',
                        'LLM_Merge_Tokens', 'LLM_Total_Tokens', 'LLM_Invoke', 'LLM_Merge_Invoke', 'LLM_Total_Invoke', 'MatchCount', 'TotalCount', 'MatchRatio'])

    return result_file


def correct_template_general(template):
    # Substitute consecutive variables only if separated with any delimiter including "." (DV)
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    # Substitute consecutive variables only if not separated with any delimiter including space (CV)
    # NOTE: this should be done at the end
    # print("CV: ", template)
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        template = re.sub(r'<\*>\:<\*>', '<*>', template)
        template = re.sub(r'<\*> <\*>', '<*>', template)
        if prev == template:
            break
    # while "<*>:<*>" in template:
    #     template = template.replace("<*>:<*>", "<*>")

    return template


def align_with_null_values(groudtruth_row):
    """
    Align the null values in the groundtruth with Content.
    """

    log = groudtruth_row['Content']
    template = groudtruth_row['EventTemplate']

    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"
    matches = re.search(regex, log)

    if matches == None:
        return template

    parts = []
    for index, part in enumerate(template.split("<*>")):
        parts.append(part)
        if index < len(matches.groups()):
            if matches.groups()[index] == '':
                parts.append('')
            else:
                parts.append('<*>')
    return ''.join(parts)


def is_file_empty(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        return len(content) == 0


def evaluator(
        dataset,
        input_dir,
        output_dir,
        log_file,
        result_file,
        lstm=False
):
    """
    Unit function to run the evaluation for a specific configuration.

    """

    print('\n=== Evaluation on %s ===' % dataset)
    indir = os.path.join(input_dir, os.path.dirname(log_file))
    log_file_basename = os.path.basename(log_file)
    groundtruth = os.path.join(indir, log_file_basename + '_structured.csv')

    parsedresult = os.path.join(
        output_dir, log_file_basename + '_structured.csv')

    # if not os.path.exists(parsedresult):
    #     with open(parsedresult, 'w') as fw:
    #         pass
    print(parsedresult)
    if not os.path.exists(parsedresult) or is_file_empty(parsedresult):
        print("No output file generated.")
        result = dataset + ',' + \
            "None" + ',' + \
            "None" + ',' + \
            "None" + ',' + \
            "None" + ',' + \
            "None" + ',' + \
            "None" + ',' + \
            "None" + ',' + \
            "None" + ',' + \
            "None" + ',' + \
            "None" + '\n'
        # "{:.1f}".format(GA_end_time) + ',' + \
        # "{:.1f}".format(PA_end_time) + ',' + \
        # "{:.1f}".format(TA_end_time) + ',' + \

        with open(os.path.join(output_dir, result_file), 'a') as summary_file:
            summary_file.write(result)
        return

    parsedresult = pd.read_csv(parsedresult, dtype=str)
    parsedresult.fillna("", inplace=True)
    groundtruth = pd.read_csv(groundtruth, dtype=str)
    # print("Start to modify output")
    # parsedresult['EventTemplate'] = parsedresult['EventTemplate'].apply(lambda x: correct_single_template(x))
    # groundtruth['EventTemplate'] = groundtruth['EventTemplate'].apply(lambda x: correct_single_template(x))

    # remove null values
    tqdm.pandas()
    print("Start to align with null values")
    groundtruth['EventTemplate'] = groundtruth.progress_apply(
        align_with_null_values, axis=1)
    groundtruth['EventTemplate'] = groundtruth['EventTemplate'].map(
        correct_template_general)
    parsedresult['EventTemplate'] = parsedresult.progress_apply(
        align_with_null_values, axis=1)

    filter_templates = None
    print("Start compute grouping accuracy")
    # calculate grouping accuracy
    start_time = time.time()
    GA, FGA = evaluate(groundtruth, parsedresult, filter_templates)

    GA_end_time = time.time() - start_time
    print('Grouping Accuracy calculation done. [Time taken: {:.3f}]'.format(
        GA_end_time))

    # calculate parsing accuracy
    start_time = time.time()
    if lstm == True:
        PA = calculate_parsing_accuracy_lstm(
            groundtruth, parsedresult, filter_templates)
        print("Finish calculate_parsing_accuracy_lstm")
    else:
        PA = calculate_parsing_accuracy(
            groundtruth, parsedresult, filter_templates)
    PA_end_time = time.time() - start_time
    print('Parsing Accuracy calculation done. [Time taken: {:.3f}]'.format(
        PA_end_time))

    # calculate template-level accuracy
    start_time = time.time()
    if lstm == True:
        tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level_lstm(
            dataset, groundtruth, parsedresult, filter_templates)
    else:
        tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level(
            dataset, groundtruth, parsedresult, filter_templates)
    TA_end_time = time.time() - start_time
    print(
        'Template-level accuracy calculation done. [Time taken: {:.3f}]'.format(TA_end_time))

    # read the time cost
    time_cost_file = os.path.join(output_dir, 'time_cost.json')
    parsing_time, training_time = 0, 0
    if os.path.exists(time_cost_file):
        with open(time_cost_file, 'r') as file:
            time_table = json.load(file)
            training_time = time_table[dataset].get('TrainingTime', 0)
            parsing_time = time_table[dataset].get('ParsingTime', 0)
            cache_time = time_table[dataset].get('CacheTime', 0)
            pretrained_model_time = time_table[dataset].get(
                'PreTrainedModelTime', 0)
            LLM_time = time_table[dataset].get('LLM_Time', 0)
            LLM_merge_time = time_table[dataset].get('LLM_Merge_Time', 0)
            LLM_total_time = time_table[dataset].get('LLM_Total_Time', 0)
            LLM_tokens = time_table[dataset].get('LLM_Tokens', 0)
            LLM_merge_tokens = time_table[dataset].get('LLM_Merge_Tokens', 0)
            LLM_total_tokens = time_table[dataset].get('LLM_Total_Tokens', 0)
            LLM_invoke = time_table[dataset].get('LLM_Invoke', 0)
            LLM_merge_invoke = time_table[dataset].get('LLM_Merge_Invoke', 0)
            LLM_total_invoke = time_table[dataset].get('LLM_Total_Invoke', 0)
            match_count = time_table[dataset].get('MatchCount', 0)
            total_count = time_table[dataset].get('TotalCount', 0)
            match_ratio = time_table[dataset].get('MatchRatio', 0)

    result = dataset + ',' + \
        "{:.3f}".format(training_time) + ',' + \
        "{:.3f}".format(parsing_time) + ',' + \
        str(tool_templates) + ',' + \
        str(ground_templates) + ',' + \
        "{:.3f}".format(GA) + ',' + \
        "{:.3f}".format(FGA) + ',' + \
        "{:.3f}".format(PA) + ',' + \
        "{:.3f}".format(FTA) + ',' + \
        "{:.3f}".format(PTA) + ',' + \
        "{:.3f}".format(RTA) + ',' + \
        "{:.3f}".format(cache_time) + ',' + \
        "{:.3f}".format(pretrained_model_time) + ',' + \
        "{:.3f}".format(LLM_time) + ',' + \
        "{:.3f}".format(LLM_merge_time) + ',' + \
        "{:.3f}".format(LLM_total_time) + ',' + \
        str(LLM_tokens) + ',' + \
        str(LLM_merge_tokens) + ',' + \
        str(LLM_total_tokens) + ',' + \
        str(LLM_invoke) + ',' + \
        str(LLM_merge_invoke) + ',' + \
        str(LLM_total_invoke) + ',' + \
        str(match_count) + ',' + \
        str(total_count) + ',' + \
        "{:.3f}".format(match_ratio) + '\n'
    # "{:.1f}".format(GA_end_time) + ',' + \
    # "{:.1f}".format(PA_end_time) + ',' + \
    # "{:.1f}".format(TA_end_time) + ',' + \

    with open(os.path.join(output_dir, result_file), 'a') as summary_file:
        summary_file.write(result)
