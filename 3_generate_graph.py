import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from language_model import get_retriever
from relationship_graph import RelationshipGraph

'''
Params
'''
retriever = get_retriever()
ranking_file_name = f'evidences_final_{sys.argv[1]}_{sys.argv[2]}.csv'
multigraph_file_name = f'multigraph_{sys.argv[1]}_{sys.argv[2]}.csv'
graph_file_name = f'graph_{sys.argv[1]}_{sys.argv[2]}.csv'

num_es = sys.argv[3]

'''
Start of program
'''

lock = threading.Lock() #Used to ensure thread-safe writing to CSV files

def retrieve_graph_and_write_csv(args):
    """
    Given a question ID and its evidences, extract triplets, create a DataFrame, 
    and write it to a CSV file in a thread-safe way.
    """
    qid, evidences = args
    ent1_vals, ent2_vals, rels = retriever.extract_triplets(evidences) #Extract entity relationships from evidences
    graph_with_ids = []

    #Build rows for each extracted triplet with the corresponding question ID
    for i in range(len(ent1_vals)):
        graph_with_ids.append([qid, ent1_vals[i], ent2_vals[i], rels[i]])

    multigraph_df = pd.DataFrame(graph_with_ids) #Convert to DataFrame

    #Append to CSV file
    with lock:
        multigraph_df.to_csv(graph_file_name, header=None, encoding='utf-8', mode='a', index=False)

if not os.path.isfile(multigraph_file_name):
    with open(multigraph_file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['question_id', 'entity1', 'entity2', 'relationship'])

if not os.path.isfile(graph_file_name):
    with open(graph_file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['question_id', 'entity1', 'entity2', 'relationship'])

#Load the input evidences file
evidences_df = pd.read_csv(ranking_file_name)
qid_to_evidences = dict()

#Load completed question IDs to avoid duplicated work
existing_graph_df = pd.read_csv(multigraph_file_name)
completed_qs = set(existing_graph_df['question_id'].unique())

#Organize evidences by question ID and rank
for ind, row in evidences_df.iterrows():
    qid = row['question_id']
    
    #Skip if question already processed
    if qid in completed_qs:
        continue

    evidence = row['evidence']
    ranking = row['final_rank']

    if qid not in qid_to_evidences:
        qid_to_evidences[qid] = [None for _ in range(num_es)]
        
    qid_to_evidences[qid][ranking-1] = evidence #Place evidence based on its ranking

#Build list of inputs for processing
inputs = []
for qid, evidences in qid_to_evidences.items():
    inputs.append([qid, evidences])

#Multithreading option based on command-line flag
if len(sys.argv) >= 5 and sys.argv[4] == 'multithread':
    print("Multithreading")
    with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        executor.map(retrieve_graph_and_write_csv, inputs)
else:
    for inp in inputs:
        retrieve_graph_and_write_csv(inp)

#Load updated data from CSVs
multigraph_df = pd.read_csv(multigraph_file_name)
e_df = pd.read_csv(graph_file_name)

#For each question, consolidate duplicate relationships in the graph and write simplified version
for qid in evidences_df['question_id'].unique():
    if qid in e_df['question_id'].unique():
        continue

    #Get all rows for the current question
    graph_for_q = multigraph_df[multigraph_df['question_id'] == qid]
    entity1_vals, entity2_vals, rel_vals = [], [], []

    for i, row in graph_for_q.iterrows():
        entity1_vals.append(row['entity1'])
        entity2_vals.append(row['entity2'])
        rel_vals.append(row['relationship'])

    #Build relationship graph and simplify
    graph = RelationshipGraph(retriever)
    graph.add_connections(entity1_vals, entity2_vals, rel_vals)
    graph.convert_to_simple_graph()

    #Extract simplified triplets
    entity1_vals, entity2_vals, relationships = graph.extract_triplets()

    #Write to final graph file
    graph_list = []
    for i in range(len(entity1_vals)):
        graph_list.append([qid, entity1_vals[i], entity2_vals[i], relationships[i]])
    
    graph_df = pd.DataFrame(graph_list)
    graph_df.to_csv(graph_file_name, header=None, encoding='utf-8', mode='a', index=False)
