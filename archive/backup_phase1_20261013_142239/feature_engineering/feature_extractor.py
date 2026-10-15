import pandas as pd
import re
import json
import os
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class FeatureExtractor:
    def __init__(self, config_dir: str = 'config'):
        """
        Initializes the Feature Extractor using external JSON lexicons.
        """
        # Load lexicons
        with open(os.path.join(config_dir, 'ambiguity_lexicon.json'), 'r') as f:
            ambig_dict = json.load(f)
        with open(os.path.join(config_dir, 'specificity_lexicon.json'), 'r') as f:
            spec_dict = json.load(f)
            
        self.vague_terms = self._compile_regex(ambig_dict['vague_terms'])
        self.subjective_terms = self._compile_regex(ambig_dict['subjective_terms'])
        self.approximate_quantity = self._compile_regex(ambig_dict['approximate_quantity'])
        self.undefined_reference = self._compile_regex(ambig_dict['undefined_reference'])
        self.modal_verbs = self._compile_regex(ambig_dict['modal_verbs'])
        
        self.time_units = self._compile_regex(spec_dict['time_units'])
        self.size_units = self._compile_regex(spec_dict['size_units'])
        self.capacity_values = self._compile_regex(spec_dict['capacity_values'])
        self.threshold_indicators = self._compile_regex(spec_dict['threshold_indicators'])
        self.formats_protocols = self._compile_regex(spec_dict['formats_protocols'])
        self.explicit_conditions = self._compile_regex(spec_dict['explicit_conditions'])
        
        self.measurable_number = re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b')
        self.percentage_indicator = re.compile(r'%|percent\b', re.IGNORECASE)
        self.passive_voice = re.compile(r'\b(is|are|was|were|be|been|being)\s+([a-z]+ed|built|made|done|sent)\b', re.IGNORECASE)
        
    def _compile_regex(self, word_list):
        escaped_words = [re.escape(w) for w in word_list]
        pattern = r'\b(' + '|'.join(escaped_words) + r')\b'
        return re.compile(pattern, re.IGNORECASE)

    def extract_features(self, text: str) -> dict:
        """
        Extracts structural, linguistic, and specificity features from a requirement.
        """
        if not isinstance(text, str) or not text.strip():
            return self._empty_features()
            
        # Basic Structural Features
        sentences = sent_tokenize(text)
        tokens = word_tokenize(text)
        words = [w for w in tokens if w.isalnum()]
        
        sentence_count = len(sentences)
        token_count = len(tokens)
        word_count = len(words)
        char_length = len(text)
        
        if word_count == 0:
            feats = self._empty_features()
            feats['Token_Count'] = token_count
            feats['Sentence_Count'] = sentence_count
            feats['Char_Length'] = char_length
            return feats
            
        avg_token_length = sum(len(w) for w in words) / word_count
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / word_count
        
        # Linguistic Ambiguity Features
        vague_count = len(self.vague_terms.findall(text))
        subj_count = len(self.subjective_terms.findall(text))
        approx_count = len(self.approximate_quantity.findall(text))
        undef_count = len(self.undefined_reference.findall(text))
        modal_count = len(self.modal_verbs.findall(text))
        passive_count = len(self.passive_voice.findall(text))
        
        ling_ambig_score = vague_count + subj_count + approx_count + undef_count + modal_count
        
        # Specificity Features
        num_count = len(self.measurable_number.findall(text))
        pct_count = len(self.percentage_indicator.findall(text))
        time_count = len(self.time_units.findall(text))
        size_count = len(self.size_units.findall(text))
        cap_count = len(self.capacity_values.findall(text))
        thresh_count = len(self.threshold_indicators.findall(text))
        fmt_count = len(self.formats_protocols.findall(text))
        cond_count = len(self.explicit_conditions.findall(text))
        
        # Check for comparative/range expressions
        range_count = len(re.findall(r'\b(between \d+ and \d+|\d+ to \d+)\b', text, re.IGNORECASE))
        
        # Advanced interaction: Does it have a testable metric?
        # Number + (time | size | capacity | threshold | pct)
        has_testable_metric = 1 if num_count > 0 and (time_count > 0 or size_count > 0 or cap_count > 0 or thresh_count > 0 or pct_count > 0) else 0
        if fmt_count > 0: has_testable_metric = 1
        
        spec_score = num_count + pct_count + time_count + size_count + cap_count + thresh_count + fmt_count + cond_count + range_count + has_testable_metric
        
        return {
            'Token_Count': token_count,
            'Word_Count': word_count,
            'Sentence_Count': sentence_count,
            'Char_Length': char_length,
            'Avg_Token_Length': round(avg_token_length, 2),
            'Lexical_Diversity': round(lexical_diversity, 2),
            
            'Vague_Term_Count': vague_count,
            'Subjective_Term_Count': subj_count,
            'Approximate_Quantity_Count': approx_count,
            'Undefined_Reference_Count': undef_count,
            'Modal_Verb_Count': modal_count,
            'Passive_Voice_Count': passive_count,
            'Linguistic_Ambiguity_Score': ling_ambig_score,
            
            'Numeric_Count': num_count,
            'Percentage_Count': pct_count,
            'TimeUnit_Count': time_count,
            'SizeUnit_Count': size_count,
            'Capacity_Count': cap_count,
            'Threshold_Count': thresh_count,
            'Format_Count': fmt_count,
            'Condition_Count': cond_count,
            'Range_Count': range_count,
            'Has_Testable_Metric': has_testable_metric,
            'Specificity_Score': spec_score
        }

    def _empty_features(self):
        return {
            'Token_Count': 0, 'Word_Count': 0, 'Sentence_Count': 0, 'Char_Length': 0,
            'Avg_Token_Length': 0.0, 'Lexical_Diversity': 0.0,
            'Vague_Term_Count': 0, 'Subjective_Term_Count': 0,
            'Approximate_Quantity_Count': 0, 'Undefined_Reference_Count': 0,
            'Modal_Verb_Count': 0, 'Passive_Voice_Count': 0, 'Linguistic_Ambiguity_Score': 0,
            'Numeric_Count': 0, 'Percentage_Count': 0, 'TimeUnit_Count': 0,
            'SizeUnit_Count': 0, 'Capacity_Count': 0, 'Threshold_Count': 0,
            'Format_Count': 0, 'Condition_Count': 0, 'Range_Count': 0,
            'Has_Testable_Metric': 0, 'Specificity_Score': 0
        }

    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        features = df[text_column].apply(self.extract_features).tolist()
        feat_df = pd.DataFrame(features, index=df.index)
        return pd.concat([df, feat_df], axis=1)

if __name__ == "__main__":
    fe = FeatureExtractor()
    print(fe.extract_features("The system shall process 500 requests efficiently."))
