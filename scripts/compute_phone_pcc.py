from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_api import GOPTInference
import numpy as np


def main():
    engine = GOPTInference(
        model_path="pretrained_models/gopt_librispeech/best_audio_model.pth",
        dataset_name="librispeech",
    )
    results = engine.predict_from_dataset(dataset_name="librispeech", split="test")

    preds: list[float] = []
    gts: list[float] = []
    for item in results:
        phone_pred = item["phone_scores"]
        phone_gt = item["ground_truth"]["phone_scores"]
        for p, g in zip(phone_pred, phone_gt):
            if g >= 0:
                preds.append(float(p))
                gts.append(float(g))

    preds_arr = np.asarray(preds, dtype=np.float32)
    gts_arr = np.asarray(gts, dtype=np.float32)
    pcc = np.corrcoef(preds_arr, gts_arr)[0, 1]

    print(f"Phone samples: {preds_arr.size}")
    print(f"Phone-level PCC: {pcc:.4f}")


if __name__ == "__main__":
    main()
