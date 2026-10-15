# IntelliReq — Known Limitations & Research Boundaries

## 1. Scope & Input Granularity
- **Sentence-Level Granularity:** IntelliReq is trained and evaluated on individual requirement statements (e.g. "The system shall process orders within 500 ms"). It does not perform document-level coreference resolution across multi-paragraph user stories or system manuals.
- **Minimum Length:** Requirements must contain at least 3 words. Submissions shorter than 3 words are rejected as invalid.

## 2. File Format Support Policy
- **Supported Formats:** Only `.txt` (one per line) and `.xlsx` (Excel workbook with a requirement column) are supported.
- **Disallowed Formats:** `.csv`, `.pdf`, `.docx`, and `.json` are intentionally disallowed to prevent delimiter corruption, table flattening issues, and multi-column ambiguity.

## 3. Domain Vocabulary & Acronym False Positives
- **Root Cause:** In our independent challenge evaluation, 9 out of 80 requirements (11.25%) were False Positives (specific requirements predicted as ambiguous).
- **Mechanism:** Requirements specifying modern technical acronyms without numerical bounds (e.g. `The API shall use token-based authentication via JWT`, `The backend shall be deployed on AWS EC2 instances`, `The payment module shall integrate with Stripe API v3`) have low frequency in base NLP models and lack physical units (`ms`, `GB`, `users`). The model flags them as abstract architecture statements.
- **Mitigation:** The `NEEDS REVIEW` interval correctly captures many borderline technical statements, allowing human analysts to sign off. Future work should expand the specialized software terminology vocabulary.

## 4. Short Terse False Negatives
- **Root Cause:** 3 out of 80 challenge requirements (3.75%) were False Negatives (ambiguous requirements predicted as specific).
- **Mechanism:** Extremely concise sentences such as `Access must be strictly controlled` do not contain unquantified adjectives from the regex dictionary (e.g., `fast`, `user-friendly`) and have high brevity, leading the model to underestimate ambiguity.

## 5. Suggested Rewrites
- **Heuristic Suggestions:** AI-assisted rewrites are generated using structural rule templates. They provide syntax scaffolding and metric placeholders (`[X ms]`, `[Y %]`). They do NOT invent domain constraints, business rules, or security keys. Human engineer validation is always required.
