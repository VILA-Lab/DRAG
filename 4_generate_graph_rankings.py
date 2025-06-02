import csv
import os
import sys
from collections import defaultdict

import pandas as pd

from semantic_distance_calculator import SemanticDistanceCalculator
from utils import get_qs

'''
Params
'''
dist_calc = SemanticDistanceCalculator()
graph_file_name = f'graph_{sys.argv[1]}_{sys.argv[2]}.csv'
semantic_rank_file_name = f'graph_semantic_{sys.argv[1]}_{sys.argv[2]}2.csv'
final_rank_file_name = f"graph_final_{sys.argv[1]}_{sys.argv[2]}2.csv"
benchmark_questions = get_qs([], True, True)
num_es = sys.argv[3]

'''
Start of Program
'''
graph_df = pd.read_csv(graph_file_name)
graph_df['llm_rank'] = graph_df.groupby('question_id').cumcount() + 1

#Initialize dictionaries to store relationships and their corresponding metadata
relationships_dict = defaultdict(list) #Maps question_id to a list of relationship strings
sentence_to_llm_rank = dict() #Maps a relationship string to [llm_rank, entity1, entity2]

#Populate dictionaries from graph DataFrame
for i, qid in enumerate(graph_df['question_id']):
    relationship = graph_df.iloc[i]['relationship']
    ent1 = graph_df.iloc[i]['entity1']
    ent2 = graph_df.iloc[i]['entity2']
    relationships_dict[qid].append(relationship)
    sentence_to_llm_rank[relationship] = [graph_df.iloc[i]['llm_rank'], ent1, ent2]

existing_qs = set()

if not os.path.isfile(semantic_rank_file_name):
    with open(semantic_rank_file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['question_id', 'llm_rank', 'relationship', 'entity1', 'entity2', 'semantic_rank'])

#Load existing semantic rankings to avoid duplication
existing = pd.read_csv(semantic_rank_file_name)
for q in existing['question_id'].unique():
    existing_qs.add(q)

#Compute semantic rankings for each new question
for qid, rels in relationships_dict.items():
    if qid in existing_qs:
        continue

    #Retrieve the original question text for computing semantic distance
    question = benchmark_questions.loc[benchmark_questions['id'] == qid].iloc[0]['question']

    #Rank sentences based on semantic similarity to the question
    ranked_sentences = dist_calc.get_top_k_sentences(question, rels, num_es)

    new_data = []
    for i, sentence in enumerate(ranked_sentences):
        new_data.append([
            qid,
            sentence_to_llm_rank[sentence][0],
            sentence_to_llm_rank[sentence][1],
            sentence_to_llm_rank[sentence][2],
            sentence,
            int(i+1) #semantic_rank
        ])

    new_df = pd.DataFrame(new_data)
    new_df.to_csv(semantic_rank_file_name, header=None, encoding='utf-8', mode='a', index=False)

#Load semantic rankings from file for final re-ranking
s_e_df = pd.read_csv(semantic_rank_file_name)
q_to_rank = dict() #Maps question_id to a list of combined rank info

#Combine LLM and semantic ranks
for i, row in s_e_df.iterrows():
    qid = row['question_id']

    if qid not in q_to_rank:
        q_to_rank[qid] = []

    #Combine ranks and keep all metadata for sorting
    q_to_rank[qid].append([
        row['llm_rank'] + row['semantic_rank'], #combined rank score
        row['llm_rank'],
        row['semantic_rank'],
        row['relationship'],
        row['entity1'],
        row['entity2']
    ])

new_list = []

for qid in q_to_rank:
    #Sort relationships by the sum of llm_rank and semantic_rank
    q_to_rank[qid].sort(key=lambda x: x[0])
    for i, val in enumerate(q_to_rank[qid]):
        llm_rank = val[1]
        semantic_rank = val[2]
        relationship = val[3]
        ent1 = val[4]
        ent2 = val[5]
        new_list.append([
            qid,
            i+1, #final_rank
            semantic_rank,
            llm_rank,
            relationship,
            ent1,
            ent2
        ])

final_rank_header = ['question_id', 'final_rank', 'semantic_rank', 'llm_rank', 'entity1', 'entity2', 'relationship']
new_df = pd.DataFrame(new_list)
new_df.to_csv(final_rank_file_name, header=final_rank_header, encoding='utf-8', index=False)
