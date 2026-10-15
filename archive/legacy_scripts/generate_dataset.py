import pandas as pd
import random
import uuid
import os
import hashlib

def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()[:8]

def get_challenge_set_texts():
    try:
        df = pd.read_csv('data/challenge_test_set.csv')
        texts = set(df['Requirement_Text'].str.lower().str.strip().tolist())
        return texts
    except Exception:
        return set()

def generate_controlled_pairs():
    domains = ['e-commerce', 'banking', 'healthcare', 'education', 'logistics', 'cybersecurity', 'HR', 'finance']
    
    pairs = [
        ("The system shall respond quickly.", "The system shall respond within 2 seconds for at least 95% of requests.", 'vague', 'performance'),
        ("The API must return data fast.", "The API shall return data with a maximum latency of 200 milliseconds.", 'vague', 'performance'),
        ("The application shall load the dashboard in a reasonable timeframe.", "The application shall render the dashboard within 1.5 seconds under normal load.", 'subjective', 'performance'),
        ("The database query should be efficient.", "The database query must execute in under 50 milliseconds.", 'vague', 'performance'),
        ("The application shall provide adequate availability.", "The application shall maintain 99.9% monthly availability.", 'vague', 'availability'),
        ("The system must be highly reliable.", "The system shall achieve a Mean Time Between Failures (MTBF) of at least 720 hours.", 'vague', 'reliability'),
        ("The server should rarely go down.", "The server shall have no more than 4 hours of unplanned downtime per year.", 'approximate', 'availability'),
        ("The system shall support many concurrent users.", "The system shall support 5,000 concurrent users without exceeding 80% CPU utilization.", 'approximate', 'scalability'),
        ("The database shall handle massive amounts of data.", "The database shall support up to 50 TB of structured data.", 'vague', 'capacity'),
        ("The network must accommodate heavy traffic.", "The network shall support a sustained throughput of 10 Gbps.", 'vague', 'capacity'),
        ("The system shall store data securely.", "The system shall encrypt stored customer data using AES-256.", 'vague', 'security'),
        ("The application shall provide strong authentication.", "The application shall require multi-factor authentication (MFA) via SMS or authenticator app.", 'subjective', 'security'),
        ("Passwords must be complex.", "Passwords must be at least 12 characters long and include uppercase, lowercase, numbers, and symbols.", 'vague', 'security'),
        ("The platform shall restrict access to authorized personnel.", "The platform shall enforce Role-Based Access Control (RBAC) ensuring only the 'Admin' role can delete user accounts.", 'vague', 'security'),
        ("The interface shall be easy to use.", "The interface shall allow a first-time user to complete account creation within 2 minutes without assistance.", 'subjective', 'usability'),
        ("The database shall be backed up regularly.", "The database shall be backed up every 24 hours and retain the previous 30 backups.", 'vague', 'scheduling'),
        ("The system shall periodically send reports.", "The system shall generate and send the report at 09:00 UTC on the first day of every month.", 'vague', 'scheduling'),
        ("Logs should be rotated frequently.", "Logs shall be rotated daily and archived to cold storage after 7 days.", 'vague', 'scheduling'),
        ("The system shall calculate totals accurately.", "The system shall calculate totals using double-precision floating-point arithmetic compliant with IEEE 754.", 'vague', 'accuracy'),
    ]
    
    subjects = ["The system", "The application", "The service", "The backend", "The platform", "The module", "The API", "The client app"]
    verbs = ["shall", "must", "should", "will"]
    actions = ["process", "handle", "manage", "execute", "complete"]
    vague_advs = ["quickly", "efficiently", "smoothly", "seamlessly", "adequately", "properly"]
    
    for _ in range(150):
        sub = random.choice(subjects)
        verb = random.choice(verbs)
        act = random.choice(actions)
        adv = random.choice(vague_advs)
        dom = random.choice(domains)
        num = random.choice([100, 500, 1000, 5000])
        noun = random.choice(["transactions", "requests", "records", "files", "operations"])
        
        amb_req = f"{sub} {verb} {act} {num} {noun} {adv}."
        time = random.choice(["1 second", "2 seconds", "500 milliseconds", "100 ms"])
        unamb_req = f"{sub} {verb} {act} {num} {noun} per minute with a maximum latency of {time}."
        pairs.append((amb_req, unamb_req, 'vague', dom))

    # Conditionals
    for _ in range(50):
        sub = random.choice(subjects)
        verb = random.choice(verbs)
        dom = random.choice(domains)
        amb_req = f"{sub} {verb} alert the admin when necessary."
        unamb_req = f"{sub} {verb} alert the admin via email if disk space falls below 10%."
        pairs.append((amb_req, unamb_req, 'undefined_reference', dom))
        
    records = []
    seen = set()
    challenge_texts = get_challenge_set_texts()
    
    for amb, clear, amb_type, dom in pairs:
        pair_id = f"PAIR-{get_hash(amb+clear)}"
        
        # Deduplicate and check leakage
        for req_text, label, atype in [(amb, 'Ambiguous', amb_type), (clear, 'Not Ambiguous', 'none')]:
            norm = req_text.lower().strip()
            if norm in seen or norm in challenge_texts:
                continue
            seen.add(norm)
            
            records.append({
                'id': f"REQ-{uuid.uuid4().hex[:8]}",
                'requirement': req_text,
                'label': label,
                'ambiguity_type': atype,
                'source': 'intellireq_generator',
                'source_type': 'controlled_pair',
                'domain': dom,
                'generation_method': 'synthetic_paired',
                'pair_id': pair_id,
                'original_id': '',
                'annotation_status': 'auto_generated'
            })
            
    return records

def build_adversarial_set():
    pairs = [
        ("The system shall securely store customer data.", "The system shall encrypt customer data using AES-256 at rest."),
        ("The system shall generate reports regularly.", "The system shall generate reports every Monday at 09:00."),
        ("The system shall maintain high availability.", "The system shall maintain 99.9% monthly availability."),
        ("The application shall appropriately handle errors.", "The application shall log errors to the syslog daemon and return HTTP 500."),
        ("The module shall be very efficient.", "The module shall utilize no more than 512 MB of RAM."),
        ("The system shall provide adequate capacity of 100 GB.", "The system shall provide exactly 100 GB of storage capacity per user block."),
        ("The UI shall be easy to use.", "The UI shall pass WCAG 2.1 AA compliance testing."),
        ("The backend will safely parse inputs.", "The backend will validate inputs against the JSON schema defined in spec v1.2.")
    ]
    records = []
    for amb, clear in pairs:
        records.append({'id': f"ADV-{uuid.uuid4().hex[:8]}", 'requirement': amb, 'expected_label': 'Ambiguous'})
        records.append({'id': f"ADV-{uuid.uuid4().hex[:8]}", 'requirement': clear, 'expected_label': 'Not Ambiguous'})
        
    pd.DataFrame(records).to_csv('data/adversarial_test_set.csv', index=False)
    print("Created adversarial_test_set.csv")

def build_dataset():
    print("Generating controlled pairs (leakage-checked)...")
    pair_records = generate_controlled_pairs()
    df = pd.DataFrame(pair_records)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/intellireq_clean_dataset.csv', index=False)
    
    print(f"Generated clean dataset with {len(df)} requirements.")
    build_adversarial_set()

if __name__ == "__main__":
    build_dataset()
