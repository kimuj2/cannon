import numpy as np

class InspectionModel:
    def __init__(self):
        self.model_loaded = True

    def predict(self, rgb_img: np.ndarray):
        h, w = rgb_img.shape[:2]
        mean_val = float(np.mean(rgb_img))
        std_val = float(np.std(rgb_img))

        if mean_val < 90:
            pred = "T1"
        elif mean_val < 130:
            pred = "T2"
        elif mean_val < 170:
            pred = "T3"
        else:
            pred = "T4"

        confidence = min(0.99, max(0.50, (std_val + mean_val) / 510.0))

        return {
            "ok": True,
            "class": pred,
            "confidence": round(float(confidence), 4),
            "image_size": [int(w), int(h)],
            "note": "lightweight demo model for Vercel"
        }

model_instance = InspectionModel()