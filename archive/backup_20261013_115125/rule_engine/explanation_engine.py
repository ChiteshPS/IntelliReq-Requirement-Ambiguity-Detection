from feature_engineering.feature_extractor import FeatureExtractor

class ExplanationEngine:
    def __init__(self, config_dir='config'):
        self.feature_extractor = FeatureExtractor(config_dir=config_dir)
        
    def generate_explanation(self, text: str, predicted_class: str) -> dict:
        """
        Generates an explanation and suggested rewrite based on linguistic features,
        but DOES NOT override the model's decision.
        """
        feats = self.feature_extractor.extract_features(text)
        
        reasons = []
        missing = []
        rewrite = None
        
        if predicted_class == "Not Ambiguous":
            if feats['Has_Testable_Metric'] > 0 or feats['Format_Count'] > 0 or feats['Condition_Count'] > 0:
                return {
                    "Reason": "The requirement specifies explicit measurable constraints or conditions.",
                    "Missing_Specification": [],
                    "Suggested_Rewrite": None
                }
            else:
                return {
                    "Reason": "The requirement is structurally sound and lacks obvious ambiguity indicators.",
                    "Missing_Specification": [],
                    "Suggested_Rewrite": None
                }
                
        if predicted_class == "NEEDS REVIEW":
            return {
                "Reason": "The model found conflicting evidence between the semantic context and the explicit constraints (e.g. contains both vague words and specific numbers). Manual review is required.",
                "Missing_Specification": ["Determine if the specific numbers adequately define the vague term."],
                "Suggested_Rewrite": None
            }
            
        # Ambiguous Logic
        text_lower = text.lower()
        
        vague_terms_found = self.feature_extractor.vague_terms.findall(text)
        subjective_terms_found = self.feature_extractor.subjective_terms.findall(text)
        approx_quant_found = self.feature_extractor.approximate_quantity.findall(text)
        undef_ref_found = self.feature_extractor.undefined_reference.findall(text)
        modal_verbs_found = self.feature_extractor.modal_verbs.findall(text)
        
        if vague_terms_found:
            term = list(set(vague_terms_found))[0].lower()
            if term in ["quickly", "fast"]:
                reasons.append(f'"{term}" does not specify a measurable response-time threshold.')
                missing.extend(["maximum response time", "measurement conditions"])
                rewrite = "The system shall respond within 2 seconds for at least 95% of requests."
            elif term == "securely":
                reasons.append(f'"{term}" does not specify a concrete security mechanism.')
                missing.append("specific security mechanism or encryption standard")
                rewrite = "The system shall encrypt data at rest using AES-256."
            elif term in ["periodically", "regularly"]:
                reasons.append(f'"{term}" does not define a measurable interval.')
                missing.append("specific time interval")
                rewrite = "The system shall perform a full backup every 24 hours."
            elif term in ["efficiently", "seamlessly"]:
                reasons.append(f'"{term}" does not define measurable throughput, latency, or resource limits.')
                missing.append("measurable throughput, latency, or resource limits")
                rewrite = "The system shall process at least 500 requests per second with a maximum latency of 2 seconds."
            else:
                reasons.append(f'"{term}" is vague and lacks a measurable constraint.')
                
        if subjective_terms_found:
            term = list(set(subjective_terms_found))[0].lower()
            if term in ["easy", "user-friendly", "easy to use", "easy-to-use"]:
                reasons.append(f'"{term}" is subjective and no measurable usability criterion is specified.')
                missing.append("measurable usability criterion")
                rewrite = "At least 90% of first-time users shall complete the task within 3 minutes without assistance."
            else:
                reasons.append(f'"{term}" is subjective and untestable.')
                missing.append("objective test criteria")
                
        if approx_quant_found:
            term = list(set(approx_quant_found))[0].lower()
            reasons.append(f'"{term}" does not provide an exact capacity or limit.')
            missing.append("exact quantitative limit")
            rewrite = "The system shall support 5,000 concurrent users."
            
        if undef_ref_found:
            term = list(set(undef_ref_found))[0].lower()
            reasons.append(f'"{term}" leaves the condition undefined.')
            missing.append("explicit condition or threshold")
            
        if modal_verbs_found:
            reasons.append(f"Contains optional language ('{list(set(modal_verbs_found))[0]}') rather than a mandatory constraint.")
            missing.append("mandatory constraints (use 'shall' or 'must')")
            
        if not reasons:
            reasons.append("Semantic ambiguity detected based on context, despite lack of explicit rule-based vague keywords.")
            missing.append("clarification of intent")
            
        return {
            "Reason": " ".join(reasons),
            "Missing_Specification": missing,
            "Suggested_Rewrite": rewrite
        }
