import pandas as pd
import random
import uuid
import os
import hashlib
import json

def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()[:8]

def get_existing_eval_texts():
    texts = set()
    try:
        df_c = pd.read_csv('data/challenge_test_set.csv')
        texts.update(df_c['Requirement_Text'].str.lower().str.strip().tolist())
    except: pass
    try:
        df_a = pd.read_csv('data/adversarial_test_set.csv')
        texts.update(df_a['requirement'].str.lower().str.strip().tolist())
    except: pass
    return texts

def generate_v3_dataset():
    records = []
    seen = set()
    eval_texts = get_existing_eval_texts()

    domains = ['e-commerce', 'banking', 'healthcare', 'education', 'logistics', 'cybersecurity', 'HR', 'finance', 'manufacturing', 'government']
    subjects = ["The system", "The application", "The service", "The backend", "The platform", "The module", "The API", "The client app"]
    verbs = ["shall", "must", "will"]

    pairs = []

    # A. Short but clearly Not Ambiguous
    # B. Long but Ambiguous
    for _ in range(50):
        sub = random.choice(subjects)
        dom = random.choice(domains)
        # Short Clear
        clear = f"{sub} shall use AES-256."
        # Long Ambiguous
        amb = f"In order to provide the best possible experience to our diverse user base across all supported regions and devices, {sub.lower()} should be highly intuitive and very efficient at all times."
        pairs.append((amb, clear, 'vague', dom))

    # C. Ambiguous with numbers
    # D. Ambiguous with units
    # E. Ambiguous with percentages
    for _ in range(100):
        sub = random.choice(subjects)
        dom = random.choice(domains)
        num = random.choice([2, 5, 10, 50, 100])
        unit = random.choice(['seconds', 'ms', 'GB', 'TB', 'hours'])
        pct = random.choice(['50%', '90%', '99%'])
        
        amb1 = f"{sub} shall respond in an appropriate number of {unit} for {pct} of users." # Unit and pct but ambiguous "appropriate number"
        clear1 = f"{sub} shall respond in {num} {unit} for {pct} of users."
        pairs.append((amb1, clear1, 'vague_quantity', dom))
        
        amb2 = f"{sub} shall support {num} users and be extremely reliable."
        clear2 = f"{sub} shall support {num} users with 99.9% uptime."
        pairs.append((amb2, clear2, 'vague', dom))
        
        amb3 = f"{sub} shall utilize around {num} {unit} of memory." # 'around' makes it ambiguous
        clear3 = f"{sub} shall utilize a maximum of {num} {unit} of memory."
        pairs.append((amb3, clear3, 'approximation', dom))

    # F. Clear with vague-looking words
    for _ in range(100):
        sub = random.choice(subjects)
        dom = random.choice(domains)
        
        amb = f"{sub} shall securely encrypt data."
        clear = f"{sub} shall securely encrypt data using AES-256."
        pairs.append((amb, clear, 'vague', dom))
        
        amb2 = f"{sub} shall be efficient."
        clear2 = f"{sub} shall efficiently process records at a rate of 1000 per second."
        pairs.append((amb2, clear2, 'vague', dom))
        
        amb3 = f"{sub} shall reliably deliver messages."
        clear3 = f"{sub} shall reliably deliver messages with a guaranteed exactly-once delivery semantic."
        pairs.append((amb3, clear3, 'vague', dom))

    # G. Small semantic change
    for _ in range(100):
        sub = random.choice(subjects)
        dom = random.choice(domains)
        
        amb = f"{sub} shall timeout if the request takes too long."
        clear = f"{sub} shall timeout if the request takes > 30 seconds."
        pairs.append((amb, clear, 'missing_threshold', dom))
        
        amb2 = f"{sub} shall send a notification."
        clear2 = f"{sub} shall send an email notification."
        pairs.append((amb2, clear2, 'undefined_reference', dom))

    # Add random standard diverse templates to bulk up to ~800
    for _ in range(150):
        sub = random.choice(subjects)
        dom = random.choice(domains)
        num = random.choice([10, 50, 100, 500])
        unit = random.choice(['ms', 'seconds'])
        
        amb = f"When the user clicks the button, {sub.lower()} should load the data fairly fast to ensure user satisfaction."
        clear = f"When the user clicks the button, {sub.lower()} shall load the data within {num} {unit}."
        pairs.append((amb, clear, 'subjective', dom))

    for amb, clear, amb_type, dom in pairs:
        pair_id = f"PAIR-{get_hash(amb+clear)}"
        
        for req_text, label, atype in [(amb, 'Ambiguous', amb_type), (clear, 'Not Ambiguous', 'none')]:
            norm = req_text.lower().strip()
            if norm in seen or norm in eval_texts:
                continue
            seen.add(norm)
            
            records.append({
                'id': f"REQ-{uuid.uuid4().hex[:8]}",
                'requirement': req_text,
                'label': label,
                'ambiguity_type': atype,
                'source': 'dataset_v3_generator',
                'source_type': 'controlled_pair',
                'domain': dom,
                'generation_method': 'synthetic_paired_v3',
                'pair_id': pair_id,
                'original_id': '',
                'annotation_status': 'auto_generated'
            })
            
    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/intellireq_clean_dataset_v3.csv', index=False)
    print(f"Dataset v3 generated with {len(df)} rows.")

if __name__ == "__main__":
    generate_v3_dataset()
