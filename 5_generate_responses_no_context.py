import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from language_model import get_retriever
from utils import get_qs

'''
Params
'''
small_scale_model = get_retriever()
csv_file_name = f'res_no_context_{sys.argv[1]}_{sys.argv[2]}.csv'

'''
Start of Program
'''
#Create a threading lock to ensure safe writing when multithreading
lock = threading.Lock()

#Function to generate a response using the model and write it to the CSV
def retrieve_and_write_csv(args):
    qid, question = args
    response = small_scale_model.generate_answer(question)

    evidences_with_ids = []

    evidences_with_ids.append([qid, response])
    evidences_df = pd.DataFrame(evidences_with_ids) 
    
    #Write response to CSV file (thread-safe)
    with lock:
        evidences_df.to_csv(csv_file_name, header=None, encoding='utf-8', mode='a', index=False)


header = ['question_id', 'response']

if not os.path.isfile(csv_file_name):
    with open(csv_file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([header])

qs = pd.read_csv(csv_file_name)
existing_qs = set()

for q in qs['question_id'].unique():
    existing_qs.add(q)

unanswered_questions = get_qs(existing_qs, False, False)

if len(unanswered_questions.keys()) > 0:
    inputs = []
    for k, v in unanswered_questions.items():
        inputs.append((k, v))
    print(f'Number of unanswered questions: {len(inputs)}')
    #Use multithreading if specified via command-line argument
    if len(sys.argv) >= 5 and sys.argv[4] == 'multithread':
        print("Multithreading")
        with ThreadPoolExecutor(max_workers = os.cpu_count()-5) as executor:
            executor.map(retrieve_and_write_csv, inputs)
    #Otherwise process questions sequentially
    else:
        for inp in inputs:
            retrieve_and_write_csv(inp)
