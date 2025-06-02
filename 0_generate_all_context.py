import subprocess
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--multithread", action="store_true", help="Enable multithreading")
args = parser.parse_args()

commands = [
    "python3 /root/llm-rag-retriever/1_generate_evidences.py",
    "python3 /root/llm-rag-retriever/2_generate_evidence_rankings.py",
    "python3 /root/llm-rag-retriever/3_generate_graph.py",
    "python3 /root/llm-rag-retriever/4_generate_graph_rankings.py",
]

llm = sys.argv[1]
benchmark = sys.argv[2]
num_es = sys.argv[3]

def generate_commands():
    """
    Generate all commands to be executed. The generated commands will execute the 
    Python files responsible for generating the evidences and graph relationships.

    Command line argument 1: LLM to be used
    Command line argument 2: Benchmark to be used
    Command line argument 3: Number of evidences/relationships to generate
    Optional argument --multithread: This optional argument executes multiple threads in parallel

    (see README for more details on options)
    """
    all_commands = []
    for command in commands:
        command  = f'{command} {llm} {benchmark} {num_es}'

        if args.multithread:
            command += f" multithread"

        all_commands.append(command)
    return all_commands


def execute_commands(commands):
    """
    Executs a list of commands in the command line.
    """
    for i, command in enumerate(commands, 1):
        try:
            print(f"\nExecuting command {i}/{len(commands)}: {command}")
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            print(f"Stdout:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error:\n{e.stderr}")
        except Exception as ex:
            print(f"An unexpected error occurred: {ex}")

if __name__ == "__main__":
    all_commands = generate_commands()
    execute_commands(all_commands)
