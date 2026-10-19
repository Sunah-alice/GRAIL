import os
import json
import requests
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import CLIPProcessor, CLIPModel
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# Data directory. Defaults to the repo root; override with MMQA_DATA_DIR.
BASE_DIR = os.environ.get("MMQA_DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_FILE = os.path.join(BASE_DIR, 'MMQA_images.jsonl')
IMAGE_DIR = os.path.join(BASE_DIR, 'gold_images')
OUTPUT_NPY = os.path.join(BASE_DIR, 'mmqa_image_embeddings.npy')
GOLD_CHAINS_FILE = os.path.join(BASE_DIR, 'enriched_gold_chains.jsonl')
BATCH_SIZE = 32

def create_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.wikipedia.org/',
    })
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_image_url_from_wiki(session, page_url):
    """
    Extracts image URL using Wikipedia Summary API based on the page title.
    Also tries to parse HTML for infobox image if API fails.
    """
    try:
        if '/wiki/' not in page_url:
            return None
        
        page_title = page_url.split('/wiki/')[-1]
        
        # 1. Try API first (More reliable for main image)
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"
        try:
            response = session.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'originalimage' in data:
                    return data['originalimage']['source']
                if 'thumbnail' in data:
                    return data['thumbnail']['source']
        except Exception as e:
            print(f"API Error for {page_title}: {e}")

        # 2. Try HTML Parsing (Fallback)
        try:
            response = session.get(page_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Try finding infobox image
                infobox = soup.find('table', class_='infobox')
                if infobox:
                    img = infobox.find('img')
                    if img and img.get('src'):
                        src = img['src']
                        if src.startswith('//'):
                            return "https:" + src
                        elif src.startswith('/'):
                            return "https://en.wikipedia.org" + src
                        return src
                
                # Try finding thumb image
                thumb = soup.find('div', class_='thumbinner')
                if thumb:
                    img = thumb.find('img')
                    if img and img.get('src'):
                        src = img['src']
                        if src.startswith('//'):
                            return "https:" + src
                        elif src.startswith('/'):
                            return "https://en.wikipedia.org" + src
                        return src
        except Exception as e:
            print(f"HTML Parse Error for {page_url}: {e}")
            
    except Exception as e:
        print(f"General Error in get_image_url: {e}")
    return None

def download_image(args):
    item, save_dir = args
    url = item['url']
    img_id = item['id']
    path_suffix = item['path']
    save_path = os.path.join(save_dir, path_suffix)
    
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return "skipped"

    session = create_session()
    image_url = get_image_url_from_wiki(session, url)
    
    if not image_url:
        image_url = url # Fallback

    try:
        response = session.get(image_url, timeout=10, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            # Verify file size
            if os.path.getsize(save_path) < 1000:
                print(f"Downloaded file too small: {save_path} ({image_url})")
                os.remove(save_path) # Remove invalid file
                return "failed_small"
            return "downloaded"
        else:
            print(f"Failed to download {image_url}: Status {response.status_code}")
    except Exception as e:
        print(f"Exception downloading {image_url}: {e}")
        
    return "failed"

def collect_image_ids_from_gold_chains() -> set:
    """
    Collect image IDs that appear in the gold chains of enriched_gold_chains.jsonl.
    """
    img_ids = set()
    
    def _from_contexts(contexts):
        if not contexts:
            return
        for ctx in contexts:
            if isinstance(ctx, dict) and ctx.get("doc_part") == "image" and ctx.get("doc_id"):
                img_ids.add(ctx["doc_id"])
    
    if not os.path.exists(GOLD_CHAINS_FILE):
        print(f"WARNING: {GOLD_CHAINS_FILE} not found. Processing all images.")
        return None
    
    print(f"Collecting image IDs from {GOLD_CHAINS_FILE}...")
    with open(GOLD_CHAINS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                gold_chain = item.get("gold_chain", {})
                if not gold_chain:
                    continue
                
                ctype = gold_chain.get("type")
                chain = gold_chain.get("chain", {})
                
                if ctype == "compare":
                    for key in ("sub_question_1", "sub_question_2", "comparison"):
                        node = chain.get(key) if isinstance(chain, dict) else None
                        if not node:
                            continue
                        if "contexts" in node:
                            _from_contexts(node.get("contexts", []))
                        if "context" in node and isinstance(node["context"], dict):
                            _from_contexts([node["context"]])
                
                elif ctype == "intersect":
                    for key in ("question_1", "question_2", "intersection"):
                        node = chain.get(key) if isinstance(chain, dict) else None
                        if not node:
                            continue
                        if "contexts" in node:
                            _from_contexts(node.get("contexts", []))
                        if "context" in node and isinstance(node["context"], dict):
                            _from_contexts([node["context"]])
                
                elif ctype == "compose":
                    for step in chain or []:
                        ctx = step.get("context") if isinstance(step, dict) else None
                        if isinstance(ctx, dict):
                            _from_contexts([ctx])
                
                else:
                    # simple, etc.
                    for node in chain if isinstance(chain, list) else []:
                        ctx = node.get("context") if isinstance(node, dict) else None
                        if isinstance(ctx, dict):
                            _from_contexts([ctx])
            except Exception as e:
                continue
    
    print(f"Found {len(img_ids)} unique image IDs in gold chains.")
    return img_ids

def main():
    # 1. Prepare directories
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    # 2. Collect image IDs from gold chains (if available)
    gold_image_ids = collect_image_ids_from_gold_chains()
    
    # 3. Load Metadata and filter by gold chain images
    print(f"Loading metadata from {IMAGES_FILE}...")
    all_items = []
    with open(IMAGES_FILE, 'r') as f:
        for line in f:
            all_items.append(json.loads(line))
    
    # Filter to only gold chain images if available
    if gold_image_ids is not None:
        items = [item for item in all_items if item['id'] in gold_image_ids]
        print(f"Filtered to {len(items)} images from gold chains (out of {len(all_items)} total).")
    else:
        items = all_items
        print(f"Using all {len(items)} images (no gold chain filtering).")
    
    print(f"Found {len(items)} images to process.")

    # 3. Download Images
    print("Starting image downloads...")
    tasks = [(item, IMAGE_DIR) for item in items]
    
    # Check which files are missing or invalid first
    files_to_download = []
    for task in tasks:
        item, save_dir = task
        path_suffix = item['path']
        save_path = os.path.join(save_dir, path_suffix)
        if not (os.path.exists(save_path) and os.path.getsize(save_path) > 1000):
            files_to_download.append(task)
    
    print(f"Need to download {len(files_to_download)} images (skipped {len(tasks) - len(files_to_download)} existing).")

    # Use more workers for better throughput
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_image, t) for t in files_to_download]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            pass

    # 4. Compute Embeddings
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    print(f"Computing embeddings on {device}...")
    
    embeddings_dict = {}
    batch_items = []
    
    for item in tqdm(items, desc="Embedding"):
        img_path = os.path.join(IMAGE_DIR, item['path'])
        if os.path.exists(img_path) and os.path.getsize(img_path) > 1000:
            batch_items.append((item['id'], img_path))
            
        if len(batch_items) >= BATCH_SIZE:
            process_batch(model, processor, batch_items, embeddings_dict, device)
            batch_items = []
            
    if batch_items:
        process_batch(model, processor, batch_items, embeddings_dict, device)

    # 5. Save Embeddings
    print(f"Saving {len(embeddings_dict)} embeddings to {OUTPUT_NPY}...")
    np.save(OUTPUT_NPY, embeddings_dict)
    print("Done.")

def process_batch(model, processor, batch, result_dict, device):
    ids = [b[0] for b in batch]
    paths = [b[1] for b in batch]
    
    images = []
    valid_ids = []
    
    for i, path in enumerate(paths):
        try:
            image = Image.open(path).convert("RGB")
            images.append(image)
            valid_ids.append(ids[i])
        except Exception:
            continue
            
    if not images:
        return

    try:
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()
        
        for i, img_id in enumerate(valid_ids):
            result_dict[img_id] = image_features[i]
            
    except Exception as e:
        print(f"Error processing batch: {e}")

if __name__ == "__main__":
    main()
