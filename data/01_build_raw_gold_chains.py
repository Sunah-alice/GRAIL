"""
Step 1: build raw gold chains from MultimodalQA.

Reads the per-type MMQA json files (MMQA_train_*.json) and writes one
gold reasoning chain per question to raw_gold_chains.jsonl. Only Compare and
Intersect types are processed here.

The gold-chain construction logic (parsing supporting_context and
intermediate_answers for each question type) is included inline below.
"""

import os
import re
import json
import glob
from typing import List, Dict, Any

from tqdm import tqdm

# Data directory. Defaults to the repo root; override with MMQA_DATA_DIR.
BASE_DIR = os.environ.get("MMQA_DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------
# Gold-chain construction (rule-based; no external calls)
# --------------------------------------------------------------------------

def deduplicate_contexts(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dedupe contexts by (doc_id, doc_part); union row indices for tables."""
    merged = {}

    def flatten_rows(r):
        if not r:
            return []
        flat = []
        for x in r:
            if isinstance(x, list):
                flat.extend(x)
            else:
                flat.append(x)
        return flat

    for ctx in contexts:
        key = (ctx.get('doc_id'), ctx.get('doc_part'))
        if key not in merged:
            merged[key] = ctx.copy()
            if 'rows' in ctx:
                merged[key]['rows'] = set(flatten_rows(ctx['rows']))
        else:
            if ctx.get('doc_part') == 'table' and 'rows' in ctx:
                if 'rows' not in merged[key]:
                    merged[key]['rows'] = set()
                merged[key]['rows'].update(flatten_rows(ctx['rows']))

    result = []
    for ctx in merged.values():
        if 'rows' in ctx:
            ctx['rows'] = sorted(list(ctx['rows']))
        result.append(ctx)
    return result


def create_gold_chain_compose(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compose chain: one step per intermediate answer, plus the final answer."""
    chain = []
    intermediate_answers = item['metadata'].get('intermediate_answers', [])
    supporting_context = item.get('supporting_context', [])

    for step_idx, step_answers in enumerate(intermediate_answers):
        step_info = {
            'step': step_idx + 1,
            'answers': [ans['answer'] for ans in step_answers],
            'modality': step_answers[0]['modality'] if step_answers else None,
            'context': supporting_context[step_idx] if step_idx < len(supporting_context) else None
        }
        chain.append(step_info)

    final_answer = item['answers'][0]['answer']
    final_context = supporting_context[-1] if len(supporting_context) > len(intermediate_answers) else None

    chain.append({'step': 'final', 'answer': final_answer, 'context': final_context})
    return chain


def parse_pseudo_question(pseudo_q: str) -> List[str]:
    """Parse a pseudo_language_question string into a list of sub-questions."""
    if not pseudo_q:
        return []

    questions = []
    explicit_questions = re.findall(r'\{([^}]+)\}', pseudo_q)
    questions.extend(explicit_questions)
    if len(questions) >= 2:
        return questions

    intersect_match = re.search(r'when the \[([^]]+)\] was \[([^]]+)\]', pseudo_q)
    if intersect_match:
        col, val = intersect_match.groups()
        target_match = re.search(r'was the \[([^]]+)\]', pseudo_q)
        target = target_match.group(1) if target_match else "item"
        questions.append(f"Find {target} where {col} is {val}")
        return questions

    remaining_q = re.sub(r'\{[^}]+\}', '', pseudo_q)
    entity_match = re.search(r'(?:or|and)\s*\[([^]]+)\]', remaining_q)
    if not entity_match:
        entity_match = re.search(r'\[([^]]+)\]\s*(?:or|and)', remaining_q)

    if entity_match:
        questions.append(entity_match.group(1))
    else:
        candidates = re.findall(r'\[([^]]+)\]', remaining_q)
        for cand in reversed(candidates):
            if '|' not in cand and 'Table' not in cand:
                questions.append(cand)
                break
    return questions


def extract_contexts(answers: List[Dict[str, Any]], table_id: str) -> List[Dict[str, Any]]:
    """Collect contexts (text/image instances and table row indices) from answers."""
    contexts = []
    for ans in answers:
        if ans.get('text_instances'):
            for inst in ans['text_instances']:
                contexts.append({'doc_id': inst['doc_id'], 'doc_part': inst['part']})
        if ans.get('image_instances'):
            for inst in ans['image_instances']:
                contexts.append({'doc_id': inst['doc_id'], 'doc_part': inst['doc_part']})
        if ans.get('table_indices'):
            contexts.append({'doc_id': table_id, 'doc_part': 'table', 'rows': ans['table_indices']})
    return deduplicate_contexts(contexts)


def create_gold_chain_compare(item: Dict[str, Any]) -> Dict[str, Any]:
    """Compare chain: two sub-questions plus a final comparison step."""
    intermediate_answers = item['metadata'].get('intermediate_answers', [])
    final_answer = item['answers'][0]['answer']
    table_id = item['metadata'].get('table_id')

    subq1_contexts = extract_contexts(intermediate_answers[0], table_id) if len(intermediate_answers) > 0 else []
    subq2_contexts = extract_contexts(intermediate_answers[1], table_id) if len(intermediate_answers) > 1 else []
    final_contexts = extract_contexts(item['answers'], table_id)

    return {
        'sub_question_1': {
            'answers': [ans['answer'] for ans in intermediate_answers[0]] if len(intermediate_answers) > 0 else [],
            'contexts': subq1_contexts
        },
        'sub_question_2': {
            'answers': [ans['answer'] for ans in intermediate_answers[1]] if len(intermediate_answers) > 1 else [],
            'contexts': subq2_contexts
        },
        'comparison': {
            'answer': final_answer,
            'operation': 'compare',
            'context': final_contexts[0] if final_contexts else None
        }
    }


def create_gold_chain_intersect(item: Dict[str, Any]) -> Dict[str, Any]:
    """Intersect chain: two question branches whose answer sets intersect."""
    intermediate_answers = item['metadata'].get('intermediate_answers', [])
    final_answer = item['answers'][0]['answer']
    table_id = item['metadata'].get('table_id')

    q1_contexts = extract_contexts(intermediate_answers[0], table_id) if len(intermediate_answers) > 0 else []
    q2_contexts = extract_contexts(intermediate_answers[1], table_id) if len(intermediate_answers) > 1 else []
    final_contexts = extract_contexts(item['answers'], table_id)

    return {
        'question_1': {
            'answer_set': [ans['answer'] for ans in intermediate_answers[0]] if len(intermediate_answers) > 0 else [],
            'contexts': q1_contexts
        },
        'question_2': {
            'answer_set': [ans['answer'] for ans in intermediate_answers[1]] if len(intermediate_answers) > 1 else [],
            'contexts': q2_contexts
        },
        'intersection': {
            'answer': final_answer,
            'operation': 'intersect',
            'context': final_contexts[0] if final_contexts else None
        }
    }


def create_gold_chain(item: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to the chain builder matching the question type."""
    qtype = item['metadata'].get('type', '')
    if qtype.startswith('Compose'):
        return {'type': 'compose', 'chain': create_gold_chain_compose(item)}
    elif qtype.startswith('Compare'):
        return {'type': 'compare', 'chain': create_gold_chain_compare(item)}
    elif qtype.startswith('Intersect'):
        return {'type': 'intersect', 'chain': create_gold_chain_intersect(item)}
    else:
        return {
            'type': 'simple',
            'chain': [{
                'answer': item['answers'][0]['answer'],
                'context': item.get('supporting_context', [])[0] if item.get('supporting_context') else None
            }]
        }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def process_all_files(output_file=None):
    """Read per-type MMQA json files and write Compare/Intersect gold chains."""
    if output_file is None:
        output_file = os.path.join(BASE_DIR, 'raw_gold_chains.jsonl')
    input_files = glob.glob(os.path.join(BASE_DIR, 'MMQA_train_*.json'))
    print(f"Found {len(input_files)} files to process.")

    total_processed = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in input_files:
            # First filter by filename (Compare / Intersect only).
            if not ('Compare' in file_path or 'intersect' in file_path.lower()):
                continue

            print(f"Processing {file_path}...")
            try:
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    data = json.load(in_f)

                for item in tqdm(data, desc=f"Processing {os.path.basename(file_path)}"):
                    try:
                        qtype = item['metadata']['type']
                        if not (qtype.startswith('Compare') or qtype.startswith('Intersect')):
                            continue

                        result = {
                            'qid': item['qid'],
                            'question': item['question'],
                            'type': item['metadata']['type'],
                            'gold_chain': create_gold_chain(item),
                        }
                        out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        total_processed += 1
                    except Exception as e:
                        print(f"Error processing item {item.get('qid', 'unknown')}: {e}")
                        continue
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

    print(f"\nDone. {total_processed} Compare/Intersect items saved to {output_file}")


if __name__ == "__main__":
    process_all_files()
