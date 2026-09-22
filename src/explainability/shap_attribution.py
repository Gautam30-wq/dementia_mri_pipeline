import shap
import numpy as np

def generate_shap_explanation(model, input_features: np.ndarray, feature_names: list, target_class: int) -> list:
    """
    Generates SHAP values for the specified target class and applies a monotonic stopping rule[cite: 2].
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_features)
    
    # Extract SHAP values for the specific class (handling single or multi-class outputs)
    if isinstance(shap_values, list):
        class_shap = shap_values[target_class][0]
    else:
        class_shap = shap_values[0]
        
    # Sort features by absolute SHAP magnitude (importance)
    importance_order = np.argsort(np.abs(class_shap))[::-1]
    
    ranked_features = []
    for idx in importance_order:
        ranked_features.append({
            'feature': feature_names[idx],
            'shap_value': class_shap[idx],
            'magnitude': np.abs(class_shap[idx])
        })
        
    # --- Monotonic Stopping Rule ---
    # Always report top 3. Report 4th/5th only if magnitude >= 50% of the previous feature[cite: 2].
    final_report = ranked_features[:3]
    for i in range(3, min(5, len(ranked_features))):
        if ranked_features[i]['magnitude'] >= 0.5 * ranked_features[i-1]['magnitude']:
            final_report.append(ranked_features[i])
        else:
            break
            
    return final_report