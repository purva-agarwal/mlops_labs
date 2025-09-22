import joblib
import os
from utils import get_project_root

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model_path = os.path.join(get_project_root(), "model", "wine_model.pkl")
    model_path = os.path.normpath(model_path)
    model = joblib.load(model_path)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    class_labels = model.classes_  # e.g., [0, 1, 2]
    display_labels = ["Wine Type A", "Wine Type B", "Wine Type C"]
    label_map = {k: v for k, v in zip(class_labels, display_labels)}

    # Final predicted label and confidence
    pred_class = y_pred[0]
    pred_label = label_map.get(pred_class, str(pred_class))
    confidence = float(y_prob[0].max())
    confidence_str = f"{confidence * 100:.1f}%"

    # Build dictionary of all class probabilities
    probabilities = {
        label_map[class_idx]: f"{prob * 100:.1f}%"
        for class_idx, prob in zip(class_labels, y_prob[0])
    }

    return pred_label, confidence_str, probabilities