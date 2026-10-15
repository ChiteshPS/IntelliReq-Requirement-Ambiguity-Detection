import pandas as pd
import random
import uuid
import os
import re

def load_challenge_texts():
    texts = set()
    try:
        df = pd.read_csv('data/challenge_test_set.csv')
        texts.update(df['Requirement_Text'].str.lower().str.strip().tolist())
    except: pass
    try:
        df2 = pd.read_csv('data/adversarial_test_set.csv')
        texts.update(df2['requirement'].str.lower().str.strip().tolist())
    except: pass
    return texts

def generate_v4():
    records = []
    seen = set()
    eval_texts = load_challenge_texts()
    
    # 1. Base Real Dataset (Natural Language)
    try:
        df_base = pd.read_csv('software_requirements_extended.csv')
        amb_words = ["may", "might", "could", "approximately", "typically", "usually", "around", "possible", "easy", "efficient", "quick", "user-friendly", "as needed"]
        pattern = re.compile(r'\b(?:' + '|'.join(amb_words) + r')\b', flags=re.IGNORECASE)
        
        for _, row in df_base.iterrows():
            req = str(row.get('Requirement', ''))
            if not req.strip(): continue
            
            norm = req.lower().strip()
            if norm in seen or norm in eval_texts: continue
            
            has_amb = bool(pattern.search(req))
            has_num = bool(re.search(r'\b\d+\b', req))
            
            if has_amb and not has_num:
                label = "Ambiguous"
            elif has_num and not has_amb:
                label = "Not Ambiguous"
            else:
                continue 
                
            seen.add(norm)
            sg = f"SG-{uuid.uuid4().hex[:8]}"
            records.append({
                'id': f"REQ-{uuid.uuid4().hex[:8]}",
                'requirement': req,
                'label': label,
                'ambiguity_type': 'vague' if label == "Ambiguous" else 'none',
                'source': 'software_requirements_extended',
                'source_type': 'real',
                'domain': 'General',
                'generation_method': 'curated',
                'pair_id': sg,
                'semantic_group_id': sg,
                'annotation_status': 'auto'
            })
    except Exception as e:
        print(f"Error loading base dataset: {e}")

    # 2. Hard Negatives to break TF-IDF
    subjects = ["The system", "The module", "The API", "The server", "The UI", "The backend", "The client", "The microservice"]
    metrics = ["within 50 ms", "using AES-256", "with 99.9% uptime", "in 2 seconds", "utilizing < 1GB RAM", "returning HTTP 200", "scaling to 100 nodes"]
    amb_words = ["quickly", "efficiently", "reliably", "securely", "smoothly", "adequately", "appropriately", "seamlessly", "easy", "efficient", "quick"]
    
    # 2a. HARD NEGATIVE: Not Ambiguous (contains ambiguous word BUT ALSO a hard metric)
    for _ in range(400):
        sub = random.choice(subjects)
        amb = random.choice(amb_words)
        met = random.choice(metrics)
        req = f"{sub} shall operate {amb}, strictly defined as {met}."
        norm = req.lower().strip()
        if norm not in seen:
            seen.add(norm)
            sg = f"SG-{uuid.uuid4().hex[:8]}"
            records.append({
                'id': f"REQ-{uuid.uuid4().hex[:8]}", 'requirement': req, 'label': 'Not Ambiguous',
                'ambiguity_type': 'none', 'source': 'v4_hard_clear', 'source_type': 'synth',
                'domain': 'General', 'generation_method': 'synth', 'pair_id': sg, 'semantic_group_id': sg, 'annotation_status': 'auto'
            })

    # 2b. HARD NEGATIVE: Ambiguous (contains numbers BUT STILL vague)
    for _ in range(400):
        sub = random.choice(subjects)
        num = random.choice([2, 5, 10, 50, 100])
        unit = random.choice(["seconds", "GB", "users", "connections"])
        req = f"{sub} shall support {num} {unit} and perform adequately."
        norm = req.lower().strip()
        if norm not in seen:
            seen.add(norm)
            sg = f"SG-{uuid.uuid4().hex[:8]}"
            records.append({
                'id': f"REQ-{uuid.uuid4().hex[:8]}", 'requirement': req, 'label': 'Ambiguous',
                'ambiguity_type': 'vague', 'source': 'v4_hard_amb', 'source_type': 'synth',
                'domain': 'General', 'generation_method': 'synth', 'pair_id': sg, 'semantic_group_id': sg, 'annotation_status': 'auto'
            })

    # 2c. Length Breakers (Short Clear, Long Ambiguous)
    for _ in range(200):
        sub = random.choice(subjects)
        req_clr = f"{sub} shall log via Syslog."
        req_amb = f"In order to guarantee that the user base receives the intended experience regardless of their geographical location or network conditions, {sub.lower()} should be highly intuitive and efficient."
        
        for req, label in [(req_amb, 'Ambiguous'), (req_clr, 'Not Ambiguous')]:
            norm = req.lower().strip()
            if norm not in seen:
                seen.add(norm)
                sg = f"SG-{uuid.uuid4().hex[:8]}"
                records.append({
                    'id': f"REQ-{uuid.uuid4().hex[:8]}", 'requirement': req, 'label': label,
                    'ambiguity_type': 'various', 'source': 'v4_len_break', 'source_type': 'synth',
                    'domain': 'General', 'generation_method': 'synth', 'pair_id': sg, 'semantic_group_id': sg, 'annotation_status': 'auto'
                })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/intellireq_clean_dataset_v4.csv', index=False)
    print(f"Dataset v4 generated with {len(df)} rows.")

if __name__ == "__main__":
    generate_v4()
