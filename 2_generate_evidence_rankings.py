import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from semantic_distance_calculator import SemanticDistanceCalculator
from utils import get_qs

'''
Params
'''
dist_calc = SemanticDistanceCalculator()
evidences_file_name = f'evidences_{sys.argv[1]}_{sys.argv[2]}.csv'
semantic_rank_file_name = f'evidences_semantic_{sys.argv[1]}_{sys.argv[2]}.csv'
final_rank_file_name = f"evidences_final_{sys.argv[1]}_{sys.argv[2]}.csv"
benchmark_questions = get_qs([], True, True)

'''
Start of Program
'''
evidences_df = pd.read_csv(evidences_file_name)
evidences = defaultdict(list) #Maps question_id to list of evidence strings
sentence_to_llm_rank = dict() #Maps each evidence string to its LLM-generated relevance rank

#Populate the above data structures from the input DataFrame
for i, qid in enumerate(evidences_df['question_id']):
    evidence = evidences_df.iloc[i]['evidence']
    evidences[qid].append(evidence)
    sentence_to_llm_rank[evidence] = evidences_df.iloc[i]['evidence_relevance_rank']

#Track which questions have already been processed
existing_qs = set()

if not os.path.isfile(semantic_rank_file_name):
    with open(semantic_rank_file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['question_id', 'llm_rank', 'evidence', 'semantic_rank'])

#Read already existing question_ids from the semantic rank file to avoid reprocessing
existing = pd.read_csv(semantic_rank_file_name)
for q in existing['question_id']:
    existing_qs.add(q)

#Function to compute semantic rankings for a single question
def process_question(qid, es):
    if qid in existing_qs:
        return []

    print(qid)
    question = benchmark_questions.loc[benchmark_questions['id'] == qid].iloc[0]['question']
    
    ranked_sentences = dist_calc.get_top_k_sentences(question, es, 100)

    new_data = []
    for i, sentence in enumerate(ranked_sentences):
        new_data.append([qid, sentence_to_llm_rank[sentence], sentence, int(i + 1)])

    return new_data

new_data_list = []
if len(sys.argv) >= 5 and sys.argv[4] == 'multithread':
    print("Multithreading")
    with ThreadPoolExecutor() as executor:
        for data in executor.map(lambda qid_es: process_question(qid_es[0], qid_es[1]), evidences.items()):
            new_data_list.extend(data)
else:
    for qid, es in evidences.items():
        data = process_question(qid, es)
        new_data_list.extend(data)


new_df = pd.DataFrame(new_data_list)
new_df.to_csv(semantic_rank_file_name, header=None, encoding='utf-8', mode='a', index=False)

#Load semantic rank data for final ranking step
s_e_df = pd.read_csv(semantic_rank_file_name)
q_to_rank = defaultdict(list) #Maps question_id to list of combined rank info

#Compute combined ranks (llm_rank + semantic_rank)
for i, row in s_e_df.iterrows():
    qid = row['question_id']
    q_to_rank[qid].append([
        row['llm_rank'] + row['semantic_rank'], #Combined score
        row['llm_rank'],
        row['semantic_rank'],
        row['evidence']
    ])

#Final re-ranking based on combined score
new_list = []
for qid in q_to_rank:
    #Sort evidences by combined rank
    q_to_rank[qid].sort(key=lambda x: x[0])
    for i, val in enumerate(q_to_rank[qid]):
        llm_rank = val[1]
        semantic_rank = val[2]
        evidence = val[3]
        new_list.append([qid, i + 1, semantic_rank, llm_rank, evidence])

final_rank_header = ['question_id', 'final_rank', 'semantic_rank', 'llm_rank', 'evidence']
new_df = pd.DataFrame(new_list)
new_df.to_csv(final_rank_file_name, header=final_rank_header, encoding='utf-8', index=False)
