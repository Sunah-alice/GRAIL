
import os
import json
from tqdm import tqdm

# Data directory. Defaults to the repo root; override with MMQA_DATA_DIR.
BASE_DIR = os.environ.get("MMQA_DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEXTS_FILE = os.path.join(BASE_DIR, 'MMQA_texts.jsonl')
TABLES_FILE = os.path.join(BASE_DIR, 'MMQA_tables.jsonl')
IMAGES_FILE = os.path.join(BASE_DIR, 'MMQA_images.jsonl')

INPUT_FILE = os.path.join(BASE_DIR, 'raw_gold_chains.jsonl')
OUTPUT_FILE = os.path.join(BASE_DIR, 'enriched_gold_chains.jsonl')


def load_resources():
    """
    Load text/table/image resources into memory for doc_id lookup.
    """
    resources = {
        'texts': {},
        'tables': {},
        'images': {}
    }
    
    # Load texts.
    if os.path.exists(TEXTS_FILE):
        print(f"Loading Texts from {TEXTS_FILE}...")
        with open(TEXTS_FILE, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Texts"):
                item = json.loads(line)
                resources['texts'][item['id']] = item['text']
    else:
        print(f"Warning: {TEXTS_FILE} not found")

    # Load tables.
    if os.path.exists(TABLES_FILE):
        print(f"Loading Tables from {TABLES_FILE}...")
        with open(TABLES_FILE, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Tables"):
                item = json.loads(line)
                resources['tables'][item['id']] = item
    else:
        print(f"Warning: {TABLES_FILE} not found")

    # Load images.
    if os.path.exists(IMAGES_FILE):
        print(f"Loading Images from {IMAGES_FILE}...")
        with open(IMAGES_FILE, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Images"):
                item = json.loads(line)
                resources['images'][item['id']] = {
                    'path': item['path'],
                    'title': item['title'],
                    'url': item['url']
                }
    else:
        print(f"Warning: {IMAGES_FILE} not found")
        
    return resources

def enrich_context(context, resources):
    """
    Enrich a single context item (doc_id, doc_part) with its actual content.
    """
    doc_id = context.get('doc_id')
    doc_part = context.get('doc_part')
    
    # Keep the base structure.
    enriched = {
        'doc_id': doc_id,
        'doc_part': doc_part
    }
    
    if doc_part == 'text':
        if doc_id in resources['texts']:
            enriched['content'] = resources['texts'][doc_id]
        else:
            enriched['content'] = None
            
    elif doc_part == 'table':
        if doc_id in resources['tables']:
            table_data = resources['tables'][doc_id]
            
            # Structure the table: header + rows.
            header = [h['column_name'] for h in table_data['table'].get('header', [])]
            
            # Extract all rows.
            all_rows = []
            for row in table_data['table'].get('table_rows', []):
                # Take each cell's 'text' only.
                all_rows.append([cell['text'] for cell in row])
            
            # Keep only the referenced rows if row indices are given.
            target_rows = []
            row_indices = context.get('rows')
            
            if row_indices:
                for idx in row_indices:
                    if 0 <= idx < len(all_rows):
                        target_rows.append(all_rows[idx])
            else:
                # No row info: include all rows (backward compatible).
                target_rows = all_rows

            enriched['content'] = {
                'header': header,
                'rows': target_rows,
                'title': table_data.get('title', ''),
                'url': table_data.get('url', '')
            }
            # Keep the original row indices too.
            if row_indices:
                enriched['rows'] = row_indices
        else:
            enriched['content'] = None
            
    elif doc_part == 'image':
        if doc_id in resources['images']:
            img_info = resources['images'][doc_id]
            enriched['content'] = {
                'path': img_info['path'],
                'title': img_info['title'],
                'url': img_info['url']
            }
        else:
            enriched['content'] = None
            
    return enriched

def enrich_gold_chain(chain, resources):
    """
    Walk the gold chain and enrich its contexts.
    """
    new_chain = chain.copy()
    
    # Branch helper.
    def process_branch(branch):
        if 'contexts' in branch:
            new_contexts = []
            for ctx in branch['contexts']:
                new_contexts.append(enrich_context(ctx, resources))
            branch['contexts'] = new_contexts
        return branch

    if 'chain' in new_chain:
        c = new_chain['chain']
        
        # Compare Type
        if 'sub_question_1' in c:
            c['sub_question_1'] = process_branch(c['sub_question_1'])
        if 'sub_question_2' in c:
            c['sub_question_2'] = process_branch(c['sub_question_2'])
        if 'comparison' in c and 'context' in c['comparison'] and c['comparison']['context']:
             c['comparison']['context'] = enrich_context(c['comparison']['context'], resources)
             
        # Intersect Type
        if 'question_1' in c:
            c['question_1'] = process_branch(c['question_1'])
        if 'question_2' in c:
            c['question_2'] = process_branch(c['question_2'])
        if 'intersection' in c and 'context' in c['intersection'] and c['intersection']['context']:
            c['intersection']['context'] = enrich_context(c['intersection']['context'], resources)
            
    return new_chain

def main():
    print("Starting enrichment process...")
    
    # Load resources.
    resources = load_resources()
    
    print(f"Processing {INPUT_FILE} -> {OUTPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found!")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        count = 0
        for line in tqdm(f_in, desc="Enriching Chains"):
            try:
                item = json.loads(line)
                
                if 'gold_chain' in item:
                    item['gold_chain'] = enrich_gold_chain(item['gold_chain'], resources)
                
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
            
    print(f"Done! {count} items saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
