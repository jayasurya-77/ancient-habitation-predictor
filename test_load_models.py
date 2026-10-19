import contextlib, io, json
from pathlib import Path
from catboost import CatBoostClassifier
import tensorflow as tf

p = Path(__file__).resolve().parent

try:
    m = CatBoostClassifier()
    m.load_model(str(p / "catboost_model.cbm"))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        pollen = tf.keras.models.load_model(
            str(p / "pollen_model.keras"),
            compile=False,
            safe_mode=False
        )
    mapping = json.loads((p / "pollen_class_mapping.json").read_text(encoding="utf-8"))
    print("CatBoost: OK")
    print("Pollen model: OK")
    print(f"Class mappings: {len(mapping)}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")