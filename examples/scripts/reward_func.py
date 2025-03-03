import torch
import re
import json
import os

PATH = '/kaggle/working/OpenRLHF/batch_solution'
def reward_func(queries, prompts, labels):
    if os.path.exists(PATH):
        cur_i = max([b.split('.json')[0] for b in os.listdir('batch_solution') if b.endswith('.json')])
        cur_i = int(cur_i) + 1
        json.dump(list(zip(queries, labels)),open(f'{PATH}/{cur_i}.json','w'))
    # queries is prompts + responses
    # labels is answers
    rewards = [reward_math_func(q, l) for q, l in zip(queries, labels)]
    return torch.tensor(rewards, dtype=torch.float)


def extract_numerical_answer(answer):
    """Extract the numerical answer enclosed in \boxed{}"""
    match = re.search(r'\\boxed\{([^}]*)\}', answer)
    if match:
        extracted_answer = match.group(1)
        extracted_answer = re.sub(r'[^0-9.]', '', extracted_answer).strip()
        if extracted_answer.replace('.', '', 1).isdigit():
            return extracted_answer
    return None

def reward_math_func(answer, numerical_solution):
    extracted_answer = extract_numerical_answer(answer)
    if extracted_answer is None:
        return 0
    return int(float(numerical_solution) == float(extracted_answer))
