import os
import re
import sys
import time
from abc import ABC, abstractmethod

import anthropic
import google.generativeai as genai
import requests
from dotenv import load_dotenv
from groq import Groq
from llamaapi import LlamaAPI
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import clean_list, clean_string

# NOTE: Change MAX_RETRIES based on max number of attempts for calling APIs
MAX_RETRIES = 40

load_dotenv()
num_es = sys.argv[3]

class LanguageModel(ABC):
    """
    Abstract base class for language model-based reasoning pipelines.
    Provides a structure for interacting with LLM APIs to retrieve evidence,
    extract entity relationships, find relevant entities, combine relationships,
    and generate answers.

    Subclasses must implement the `call_api` method, which interfaces with the actual language model.
    """

    @abstractmethod
    def call_api(self, prompt, system_message=None):
        """
        Abstract method to call the underlying language model API.

        Args:
            prompt (str): The prompt to send to the model.
            system_message (str, optional): An optional system-level instruction.

        Returns:
            str: The response from the language model.
        """
        pass

    def retrieve_evidences(self, question):
        """
        Retrieves a fixed number of textual evidences relevant to answering the given question.
        Evidence is ordered from most to least relevant.
        """
        
        system_message = '''You are an assistant in charge of generating factual evidences that aid in solving the provided question.
        Provide only the evidences with no additional remarks. Do not give the answer away directly in the evidence.
        '''
        retrieve_evidence_prompt = f'''Generate {num_es} evidences that pertain to answering the following question: {question}
        The evidences must be ordered from most relevant to least relevant to answering the question.
        Separate each evidence with '$$$'.
        '''
        evidences = []

        while len(evidences) != num_es:
            evidences = self.call_api(retrieve_evidence_prompt, system_message)
            evidences = evidences.split("$$$")
            evidences = [x.strip() for x in evidences if len(x) > 1]
            evidences = clean_list(evidences)

            if len(evidences) > num_es:
                evidences = evidences[:num_es]
        return evidences

    def extract_triplets(self, evidences):
        """
        Extracts entity-relationship triplets from a list of evidences using the language model.
        """
        
        system_message = f'''You are an assistant in charge of extracting entities and entity relationships from various statements. 
        For each statement provided, extract the most important relationship between two entities in the statement.
        In total there must be {num_es} relationships extracted.
        You must end each entry with $$$.
        For example, for the statement "Sagittarius A* aligns with the dynamical center of the galaxy", the output should be:
        Entity 1:Sagittarius A*
        Entity 2: galaxy
        Relationship: Sagittarius aligns with the dynamical center of the galaxy
        $$$
        '''

        extract_triplets_prompt = f'''
        Statements: {evidences}
        '''
        
        entity1_indices = []
        entity2_indices = []
        relationship_indices = []

        while len(entity1_indices) != num_es or len(entity2_indices) != num_es or len(relationship_indices) != num_es:
            response = self.call_api(extract_triplets_prompt, system_message)
            entity1_indices = [m.start() for m in re.finditer('Entity 1:', response)]
            entity2_indices = [m.start() for m in re.finditer('Entity 2:', response)]
            relationship_indices = [m.start() for m in re.finditer('Relationship:', response)]

        entity1_vals = []
        entity2_vals = []
        relationships = []

        for i in range(len(entity1_indices)):
            ent1_ind = entity1_indices[i]
            ent2_ind = entity2_indices[i]
            rel_ind = relationship_indices[i]
            entity1 = response[ent1_ind+len('Entity 1:'):ent2_ind]
            entity2 = response[ent2_ind+len('Entity 2:'):rel_ind]

            if i != len(entity1_indices) - 1:
                rel = response[rel_ind+len('Relationship:'):entity1_indices[i+1]]
            else:
                rel = response[rel_ind+len('Relationship:'):]

            entity1_vals.append(entity1.strip())
            entity2_vals.append(entity2.strip())
            relationships.append(rel.strip())

        entity1_vals = clean_list(entity1_vals)
        entity2_vals = clean_list(entity2_vals)
        relationships = clean_list(relationships)

        return entity1_vals, entity2_vals, relationships

    def find_relevant_entities(self, question, all_entities):
        """
        Identifies which entities from a list are mentioned in a question.
        """
        
        system_message = '''You are in charge of finding all entities in the provided entity list that are mentioned in the provided question.
        You must separate each value in your response with a comma.
        '''

        prompt = f'''Question: {question}
        Entity List: {all_entities}
        '''
        response = self.call_api(prompt, system_message)
        relevant_entities = response.split(',')
        return relevant_entities

    def combine_relationships(self, relationships):
        """
        Combines multiple relationship statements into a concise, coherent summary.
        """
        
        system_message = '''You are an assistant in charge of combining the provided statements into one summarized statement. Be concise without losing any of the information.
        '''
        prompt = f'''Statements: {relationships}'''
        response = self.call_api(prompt, system_message)
        return response

    def generate_answer(self, question):
        """
        Generates a direct answer to the question using the language model.
        """
       
        system_message = '''You are a teacher in charge of correctly answering questions.
        '''
        prompt = f'''Question: {question}'''
        response = self.call_api(prompt, system_message)
        return response



class GeminiRetriever(LanguageModel):
    def __init__(self, model='gemini-1.5-flash'):
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.model = genai.GenerativeModel(model)
    
    def call_api(self, prompt, system_message):
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                attempt += 1
                time.sleep(10)
        print(f"Failed to get chat completion after {MAX_RETRIES} attempts")
        return None


class GPTRetriever(LanguageModel):
    def __init__(self, model='gpt-3.5-turbo'):
        self.client = OpenAI(api_key=os.environ['OPENAI_KEY'])
        self.model=model

    def call_api(self, prompt, system_message):
        client = self.client

        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]

        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                response = client.chat.completions.create(messages=messages,
                                                            model=self.model)
                return response.choices[0].message.content
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                attempt += 1
                time.sleep(10)
        print(f"Failed to get chat completion after {MAX_RETRIES} attempts")
        return None

class GroqRetriever(LanguageModel):
    '''
    Using Llama API from https://console.groq.com/docs/text-chat
    Available models: https://console.groq.com/docs/models 
    '''
    def __init__(self, model='llama3-8b-8192'):
        self.client = Groq(
            api_key=os.environ['GROQ_KEY'],
        )
        self.model = model

    def call_api(self, prompt, system_message):
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]

        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model
                
                )
                return response.choices[0].message.content

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                attempt += 1
                time.sleep(10)
        print(f"Failed to get chat completion after {MAX_RETRIES} attempts")
        return None

class LlamaRetriever(LanguageModel):
    def __init__(self, model='llama3.3-70b'):
        self.model = model
        self.client = LlamaAPI(os.environ['LLAMA_KEY'])
    
    def call_api(self, prompt, system_message):
        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "user", 
                    "content": f'{system_message}\n{prompt}'
                },
            ],
        }
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                response = self.client.run(request)
                text_response = response.json()['choices'][0]['message']['content']
                return clean_string(text_response)
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                attempt += 1
                time.sleep(10)
        


class LlamaWithWeightsRetriever(LanguageModel):
    def __init__(self, model='Llama-2-7b-chat-hf'):
        self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{model}")
        self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{model}")

    def call_api(self, prompt, system_message):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids=inputs['input_ids'],
            temperature=0.7, # lower = more deterministic
            num_beams=5 # higher = better quality, slower
        )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text


class ClaudeRetriever(LanguageModel):
    def __init__(self, model=''):
        self.client = anthropic.Anthropic(api_key=os.environ['CLAUDE_KEY'])
    
    def call_api(self, prompt, system_message):
        attempt = 0

        while attempt < MAX_RETRIES:
            try:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    temperature=0,
                    system=system_message,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                )
                return clean_string(response.content[0].text)
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                attempt += 1
                time.sleep(10)
        print(f"Failed to get chat completion after {MAX_RETRIES} attempts")
        return None

class DeepSeekRetriever(LanguageModel):
    def __init__(self, model='deepseek-chat'):
        self.client = OpenAI(api_key=os.environ['DEEPSEEK_KEY'], base_url="https://api.deepseek.com")
        self.model = model
        
    def call_api(self, prompt, system_message):
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False
                )

                response_text = response.choices[0].message.content

                return clean_string(response_text)
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                attempt += 1
                time.sleep(10)
        print(f"Failed to get chat completion after {MAX_RETRIES} attempts")
        return None


def get_retriever():
    """
    Returns an instance of a retriever class based on the language model name provided as a command-line argument.
    The function reads the first command-line argument (sys.argv[1]) to determine which retriever to use.
    Supported options:
        - 'gpt'      → returns GPTRetriever()
        - 'gemini'   → returns GeminiRetriever()
        - 'claude'   → returns ClaudeRetriever()
        - 'llama'    → returns LlamaRetriever()
        - 'deepseek' → returns DeepSeekRetriever()
    """
    llm_name = sys.argv[1]

    if llm_name == 'gpt':
        return GPTRetriever()
    elif llm_name == 'gemini':
        return GeminiRetriever()
    elif llm_name == 'claude':
        return ClaudeRetriever()
    elif llm_name == 'llama':
        return LlamaRetriever()
    elif llm_name == 'deepseek':
        return DeepSeekRetriever()