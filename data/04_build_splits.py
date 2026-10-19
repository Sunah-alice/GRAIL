import json
import random
import os

# Set random seed for reproducibility
random.seed(42)

# Data directory. Defaults to the repo root; override with MMQA_DATA_DIR.
BASE_DIR = os.environ.get("MMQA_DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_file = os.path.join(BASE_DIR, 'mmqa_strict_multi_hop_chains.jsonl')
output_dir = BASE_DIR

def process_data():
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    # Dictionary to store groups of samples keyed by original qid
    # content: qid -> list of generated samples
    grouped_samples = {}

    print("Reading and processing input file...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON at line {line_num + 1}: {e}")
                    continue

                qid = data.get('qid')
                question = data.get('question')
                q_type = data.get('type')
                answer = data.get('answer')
                steps = data.get('steps', [])

                if not qid or not steps:
                    continue

                # Generate samples for this chain
                # One sample for each step as target
                samples_for_this_chain = []
                for i in range(len(steps)):
                    target_step = steps[i]
                    # Context is all other steps
                    evidence_steps = steps[:i] + steps[i+1:]
                    
                    # Create the new sample object
                    new_sample = {
                        "qid": qid,
                        "question": question,
                        "question_type": q_type, # Rename 'type' to 'question_type'
                        "answer": answer,
                        "target": target_step,
                        "evidences": evidence_steps
                    }
                    samples_for_this_chain.append(new_sample)
                
                if samples_for_this_chain:
                    grouped_samples[qid] = samples_for_this_chain

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Total original questions (chains)
    all_qids = list(grouped_samples.keys())
    total_chains = len(all_qids)
    print(f"Total original chains processed: {total_chains}")

    # Shuffle the qids to randomize the split
    random.shuffle(all_qids)

    # Define split ratios (e.g., 80% train, 10% val, 10% test)
    train_ratio = 0.8
    val_ratio = 0.1
    # test_ratio = 0.1 # Implicit

    train_cutoff = int(total_chains * train_ratio)
    val_cutoff = int(total_chains * (train_ratio + val_ratio))

    train_qids = all_qids[:train_cutoff]
    val_qids = all_qids[train_cutoff:val_cutoff]
    test_qids = all_qids[val_cutoff:]

    print(f"Split sizes (chains): Train={len(train_qids)}, Val={len(val_qids)}, Test={len(test_qids)}")

    # Flatten the groups into final lists of samples
    train_samples = []
    for qid in train_qids:
        train_samples.extend(grouped_samples[qid])

    val_samples = []
    for qid in val_qids:
        val_samples.extend(grouped_samples[qid])

    test_samples = []
    for qid in test_qids:
        test_samples.extend(grouped_samples[qid])

    print(f"Total samples (lines): Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # Include original source reference if needed, but per request just "Renewal" text is fine.
    # The requirement is to not have overlapping problems (chains) between splits.
    # We achieved this by splitting on QID.

    # Function to write samples to file
    def write_jsonl(samples, filename):
        filepath = os.path.join(output_dir, filename)
        print(f"Writing {len(samples)} samples to {filepath}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

    write_jsonl(train_samples, 'mmqa_masked_renewal_train.jsonl')
    write_jsonl(val_samples, 'mmqa_masked_renewal_val.jsonl')
    write_jsonl(test_samples, 'mmqa_masked_renewal_test.jsonl')

    print("Done.")

if __name__ == "__main__":
    process_data()
