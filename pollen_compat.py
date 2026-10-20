import json
import os
import tempfile
import zipfile
from pathlib import Path
import tensorflow as tf

APP_DIR = Path(__file__).resolve().parent
POLLEN_PATH = APP_DIR / "pollen_model.keras"

def load_pollen_model():
    with zipfile.ZipFile(POLLEN_PATH, "r") as zin:
        config = json.loads(zin.read("config.json"))

        def clean(obj):
            if isinstance(obj, dict):
                obj.pop("quantization_config", None)
                for value in obj.values():
                    clean(value)
            elif isinstance(obj, list):
                for value in obj:
                    clean(value)

        clean(config)
        fd, temp_name = tempfile.mkstemp(suffix=".keras")
        os.close(fd)
        temp_path = Path(temp_name)

        try:
            with zipfile.ZipFile(temp_path, "w") as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename == "config.json":
                        data = json.dumps(config).encode("utf-8")
                    zout.writestr(item, data)

            return tf.keras.models.load_model(
                str(temp_path), compile=False, safe_mode=False
            )
        finally:
            temp_path.unlink(missing_ok=True)
