import re
import pandas as pd

class RuleBasedDetector:
    def __init__(self):
        """
        Initializes the detector with a predefined list of ambiguous words.
        These words inherently introduce uncertainty into software requirements.
        """
        # The exact list of ambiguous words requested
        self.ambiguous_words = [
            "may", "might", "could", "approximately", "typically", 
            "usually", "around", "possible", "easy", "efficient", 
            "quick", "user-friendly", "as needed"
        ]
        
        # Compile a regex pattern to match whole words only.
        # \b represents a word boundary, so we don't accidentally match "may" inside "mayor".
        # We join the words with '|' which means OR in regex.
        pattern_string = r'\b(?:' + '|'.join(self.ambiguous_words) + r')\b'
        self.regex_pattern = re.compile(pattern_string, flags=re.IGNORECASE)

    def analyze_requirement(self, text: str) -> dict:
        """
        Analyzes a single requirement string.
        Returns a dictionary with the Label and the list of matched ambiguous words.
        """
        # Find all occurrences of the ambiguous words in the text
        matches = self.regex_pattern.findall(text)
        
        # We lowercase the matches so we have a clean list 
        # (e.g. if it matched "May", we convert it to "may")
        matched_words = [match.lower() for match in matches]
        
        # Determine the label based on whether we found any matches
        if len(matched_words) > 0:
            label = "Ambiguous"
        else:
            label = "Not Ambiguous"
            
        return {
            'Rule_Label': label,
            'Matched_Ambiguous_Words': list(set(matched_words)) # Use set() to remove duplicates
        }

    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Applies the rule-based detection to an entire DataFrame.
        """
        results = df[text_column].apply(self.analyze_requirement)
        results_df = pd.DataFrame(results.tolist())
        return pd.concat([df, results_df], axis=1)

# Simple test if run directly
if __name__ == "__main__":
    detector = RuleBasedDetector()
    sample1 = "The system might need to be user-friendly."
    sample2 = "The system shall display the login screen."
    
    print(f"Sample 1: {sample1}")
    print(detector.analyze_requirement(sample1))
    print(f"\nSample 2: {sample2}")
    print(detector.analyze_requirement(sample2))
