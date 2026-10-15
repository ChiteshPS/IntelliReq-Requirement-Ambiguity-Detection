# Final Feature Specification

The IntelliReq classification pipeline uses exactly **408** features.

## 1. Sentence Embeddings (384 dimensions)
- **Model**: `all-MiniLM-L6-v2`
- **Type**: Dense Sentence Embedding
- **Purpose**: Captures semantic similarity and context, allowing the model to understand the meaning of requirements rather than just looking at isolated keywords.

## 2. Handcrafted Linguistic/Specificity Features (24 dimensions)
Extracted by `feature_engineering/feature_extractor.py`. These features quantify structural, lexical, and statistical properties of the text.

1. **Length**: Total number of words.
2. **Readability**: Flesch reading ease or similar proxy.
3. **Number of Nouns**: Using spaCy/NLTK POS tags.
4. **Number of Verbs**: Using POS tags.
5. **Number of Adjectives**: Identifies descriptive words.
6. **Number of Adverbs**: Identifies modifiers (often subjective).
7. **Number Count**: Presence of literal numeric values (e.g., '5', '100').
8. **Unit Count**: Presence of measurable units ('ms', 'GB', 'seconds').
9. **Modal Verbs**: Count of 'shall', 'must', 'should', etc.
10. **Subjective Words**: Count of known vague modifiers (e.g., 'user-friendly', 'fast').
11. **Average Word Length**: Indicates complexity.
12. **Syllable Count**: Proxy for complexity.
13. **Dependency Depth**: Maximum depth of the syntax tree.
14. **Passive Voice Count**: Count of passive verb constructions.
15. **Conjunction Count**: Number of 'and', 'or', 'but'.
16. **Preposition Count**: Number of prepositions.
17. **Pronoun Count**: Identifies unresolved references ('it', 'they').
18. **Punctuation Count**: Commas, semicolons, etc.
19. **Capitalization Ratio**: Ratio of uppercase to lowercase.
20. **Domain Keyword Match**: Match against known domain dictionaries.
21. **Action Verbs**: Verbs denoting direct action.
22. **Ambiguity Score**: Pre-calculated heuristic score based on the rule engine.
23. **Specificity Score**: Ratio of numbers/units to total words.
24. **TF-IDF Fallback**: A single dense dimension representing the maximum TF-IDF weight (or similar lexical proxy).

## Enforcement
The automated testing suite strictly enforces the resulting matrix shape to exactly `(N, 408)`. Any mismatch will fail the build gate.
