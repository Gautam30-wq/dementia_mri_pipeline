import numpy as np

class ClinicalDeferralRouter:
    """
    Routes MRI cases to either automated reporting or clinical deferral based on 
    fused confidence thresholds.
    """
    def __init__(self, confidence_threshold: float):
        self.threshold = confidence_threshold
        self.stage_names = {
            0: 'No Impairment', 
            1: 'Very Mild Impairment', 
            2: 'Mild Impairment', 
            3: 'Moderate Impairment'
        }

    def route_case(self, fused_probs: np.ndarray, fused_confidence: float) -> dict:
        """
        Determines the reporting path for a single inference case[cite: 2].
        """
        # Sort probabilities to find the top candidates
        sorted_indices = np.argsort(fused_probs)[::-1]
        top_pred = sorted_indices[0]
        second_pred = sorted_indices[1]
        
        report = {
            'confidence_score': fused_confidence,
            'is_deferred': fused_confidence < self.threshold
        }

        if not report['is_deferred']:
            # Confident Path: Auto-report single stage[cite: 2]
            report['action'] = "AUTO_REPORT"
            report['primary_diagnosis'] = self.stage_names[top_pred]
            report['explanation_targets'] = [top_pred]
        else:
            # Deferred Path: Route to clinician with differential explanation
            report['action'] = "DEFER_TO_CLINICIAN"
            report['primary_candidate'] = self.stage_names[top_pred]
            report['secondary_candidate'] = self.stage_names[second_pred]
            # Flag top two classes to generate paired Grad-CAM and SHAP explanations[cite: 2]
            report['explanation_targets'] = [top_pred, second_pred] 

        return report