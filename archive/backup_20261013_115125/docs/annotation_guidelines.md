# Annotation Guidelines: Ambiguity in Software Requirements

## 1. Primary Definition
A software requirement is **ambiguous** if it lacks objective, measurable criteria to determine whether it has been met (lack of testability) or if its phrasing allows multiple reasonable interpretations.

## 2. Context Matters
An adjective is not inherently ambiguous on its own. 

**Not Ambiguous:**
> "The system shall securely encrypt data using AES-256."
*Explanation:* While "securely" is a subjective adjective, the explicit mention of "AES-256" provides the exact testable constraint.

**Ambiguous:**
> "The system shall securely encrypt data."
*Explanation:* "Securely" is the only constraint, which is subjective. A developer has no threshold to meet.

## 3. Categories of Ambiguity

### A. Vague Adjectives and Adverbs
Terms like *fast, quick, easy, reliable, adequate, robust, efficiently, seamlessly* usually render a requirement ambiguous unless paired with a hard metric.
- *Ambiguous*: "The UI must load fast."
- *Not Ambiguous*: "The UI must load within 1.5 seconds."

### B. Unclear Quantities (Approximation)
Terms like *many, most, significant, massive, around, approximately* fail to provide testable limits.
- *Ambiguous*: "The database shall support massive amounts of data."
- *Not Ambiguous*: "The database shall support up to 50 TB of data."

### C. Undefined References or Conditions
Using pronouns or missing actors.
- *Ambiguous*: "It will alert the user when necessary." (What is "It"? When is "necessary"?)
- *Not Ambiguous*: "The scheduling service shall alert the user via email when disk space falls below 10%."

## 4. Guidelines for "Borderline" Cases
Sometimes requirements are context-dependent or rely on industry-standard interpretations (e.g. "RESTful API"). If it cannot be definitively tested as written without external assumptions, it should be marked as **Ambiguous** or flagged for **Review**.

## 5. False Positives to Avoid
Do not flag a requirement simply because it is complex or contains a word from a negative lexicon. 
- "The system shall export logs in CSV format." -> *Not Ambiguous*. "Format" does not inherently mean ambiguous.
- "The module shall utilize no more than 512 MB of RAM." -> *Not Ambiguous*. The constraint is clear.
