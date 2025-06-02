import json
import sys

import pandas as pd


def clean_string(s):
    """
    Returns a cleaned string
    """
    s = s.replace('\n', '')
    s = s.replace('\\n', '')
    
    if s.endswith('$$$'):
        s = s[:len(s) - 1]
    
    return s

def clean_list(lst):
    """
    Takes as input a list of strings, outputs cleaned strings
    """
    lst = [clean_string(val) for val in lst]
    return lst


def get_med_questions(existing_qs, include_all, as_df):
    """
    Returns MedMCQA benchmark questions
    """
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
    benchmark_questions = pd.read_parquet("hf://datasets/openlifescienceai/medmcqa/" + splits["validation"])
    
    if not include_all:
        benchmark_questions = benchmark_questions[~benchmark_questions['id'].isin(existing_qs)]
    if as_df:
        return benchmark_questions

    unanswered_questions = {}
    for ind, row in benchmark_questions.iterrows():
        question = f"{row['question']} Choices: {row['opa']}, {row['opb']}, {row['opc']}, {row['opd']}"
        unanswered_questions[row['id']] = question
    return unanswered_questions


def get_mmlu_questions(existing_qs, include_all, as_df):
    """
    Returns MMLU benchmark questions
    """
    benchmark_questions = pd.read_csv('benchmark_qs/mmlu.csv')

    if not include_all:
        benchmark_questions = benchmark_questions[~benchmark_questions['id'].isin(existing_qs)]
    
    if as_df:
        return benchmark_questions
    unanswered_questions = {}
    
    for ind, row in benchmark_questions.iterrows():
        qid = row['id']

        if qid in existing_qs:
            continue

        choices = row['choices']
        question = row['question'] + " Choices: " + choices
        unanswered_questions[row['id']] = question
    return unanswered_questions


def get_arcc_questions(existing_qs, include_all, as_df):
    """
    Returns ARC-Challenge benchmark questions
    """
    splits = {'train': 'ARC-Challenge/train-00000-of-00001.parquet', 'test': 'ARC-Challenge/test-00000-of-00001.parquet', 'validation': 'ARC-Challenge/validation-00000-of-00001.parquet'}
    benchmark_questions = pd.read_parquet("hf://datasets/allenai/ai2_arc/" + splits["train"])

    if not include_all:
        benchmark_questions = benchmark_questions[~benchmark_questions['id'].isin(existing_qs)]
    
    if as_df:
        return benchmark_questions
    unanswered_questions = {}
    
    for ind, row in benchmark_questions.iterrows():
        qid = row['id']

        if qid in existing_qs:
            continue
        choices = row['choices']
        question = row['question'] + " Choices: " + choices
        unanswered_questions[row['id']] = question
    
    return unanswered_questions


def get_fever_questions(existing_qs, include_all, as_df):
    """
    Returns FEVER 2024 benchmark questions
    """
    file_path = "benchmark_qs/fever_2024.json"

    with open(file_path, "r") as file:
        benchmark_qs = json.load(file)

    unanswered_questions = {}

    for q in benchmark_qs:
        id_val = q['id']

        if include_all == False and id_val in existing_qs:
            continue

        unanswered_questions[id_val] = 'Claim: ' + q['claim'] + '\nWhich of these 4 categories should the claim be classified into: 1. supported, 2. refuted, 3. not enough evidence, 4. conflicting evidence?'

    if as_df:
        df = pd.DataFrame(list(unanswered_questions.items()), columns=["id", "question"])
        return df
    return unanswered_questions


def get_open_leaderboard_questions(existing_qs, include_all, as_df):
    """
    Returns Open Leaderboard benchmark questions
    """
    benchmark_questions = pd.read_csv("benchmark_qs/open_leaderboard.csv")
    
    if not include_all:
        benchmark_questions = benchmark_questions[~benchmark_questions['id'].isin(existing_qs)]
    if as_df:
        return benchmark_questions

    unanswered_questions = {}
    for ind, row in benchmark_questions.iterrows():
        question = f"{row['question']}"
        unanswered_questions[row['id']] = question
    
    return unanswered_questions

def get_searchqa_questions(existing_qs, include_all, as_df):
    """
    Returns SearchQA benchmark questions
    """
    splits = {'train': 'data/train-00000-of-00001-55e7116bea868a35.parquet', 'validation': 'data/validation-00000-of-00001-2092e81367c2ca98.parquet'}
    benchmark_questions = pd.read_parquet("hf://datasets/lucadiliello/searchqa/" + splits["train"])
    benchmark_questions['id'] = benchmark_questions.index
    print(benchmark_questions)
    if not include_all:
        benchmark_questions = benchmark_questions[~benchmark_questions['id'].isin(existing_qs)]
    
    if as_df:
        return benchmark_questions
    unanswered_questions = {}
    
    for ind, row in benchmark_questions.iterrows():
        qid = row['id']

        if qid in existing_qs:
            continue
        question = row['question']
        unanswered_questions[row['id']] = question
    
    return unanswered_questions


def get_web_questions(existing_qs, include_all, as_df):
    """
    Returns Web Questions benchmark questions
    """
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    benchmark_questions = pd.read_parquet("hf://datasets/Stanford/web_questions/" + splits["train"])
    benchmark_questions['id'] = benchmark_questions.index
    print(benchmark_questions)
    if not include_all:
        benchmark_questions = benchmark_questions[~benchmark_questions['id'].isin(existing_qs)]
    
    if as_df:
        return benchmark_questions
    unanswered_questions = {}
    
    for ind, row in benchmark_questions.iterrows():
        qid = row['id']

        if qid in existing_qs:
            continue
        question = row['question']
        unanswered_questions[row['id']] = question
    
    return unanswered_questions

def get_gpqa(existing_qs, include_all, as_df=False):
    """
    Returns GPQA benchmark questions
    """
    benchmark_questions = pd.read_csv("hf://datasets/Idavidrein/gpqa/gpqa_extended.csv")
    benchmark_questions['id'] = benchmark_questions.index
    print(benchmark_questions)
    if not include_all:
        benchmark_questions = benchmark_questions[~benchmark_questions['id'].isin(existing_qs)]
    
    if as_df:
        return benchmark_questions
    unanswered_questions = {}
    
    for ind, row in benchmark_questions.iterrows():
        qid = row['id']

        if qid in existing_qs:
            continue
        question = row['question']
        unanswered_questions[row['id']] = question
    
    return unanswered_questions
    
def get_qs(existing_qs, include_all, as_df=False):
    """
    Returns benchmark questions based on command line argument
    """
    benchmark = sys.argv[2]
    if benchmark == "mmlu":
        return get_mmlu_questions(existing_qs, include_all, as_df)
    elif benchmark == "fever":
        return get_fever_questions(existing_qs, include_all, as_df)
    elif benchmark == "med":
        return get_med_questions(existing_qs, include_all, as_df)
    elif benchmark == "open":
        return get_open_leaderboard_questions(existing_qs, include_all, as_df)
    elif benchmark== "searchqa":
        return get_searchqa_questions(existing_qs, include_all, as_df)
    elif benchmark== "web":
        return get_web_questions(existing_qs, include_all, as_df)
    elif benchmark== "gpqa":
        return get_gpqa(existing_qs, include_all, as_df)
