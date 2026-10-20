
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from catboost import CatBoostClassifier

APP_DIR = Path(__file__).resolve().parent
CAT_PATH = APP_DIR / "catboost_model.cbm"
POLLEN_PATH = APP_DIR / "pollen_model.keras"
MAP_PATH = APP_DIR / "pollen_class_mapping.json"

st.set_page_config(
    page_title="Ancient Human Habitation Predictor",
    page_icon="🏛️",
    layout="wide",
)

@st.cache_resource
def get_catboost():
    model = CatBoostClassifier()
    model.load_model(str(CAT_PATH))
    return model

@st.cache_resource
def get_pollen_model():
    import tensorflow as tf

    from pollen_compat import load_pollen_model
    return load_pollen_model()

@st.cache_data
def get_mapping():
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return {i: str(v) for i, v in enumerate(data)}

    if all(str(k).isdigit() for k in data):
        return {int(k): str(v) for k, v in data.items()}

    return {int(v): str(k) for k, v in data.items()}

try:
    for path in (CAT_PATH, POLLEN_PATH, MAP_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path.name}")

    model = get_catboost()
    pollen_model = get_pollen_model()
    idx_to_pollen = get_mapping()

except Exception as e:
    st.error(f"Could not load a saved model or mapping: {type(e).__name__}: {e}")
    st.stop()

pollen_names = [idx_to_pollen[i] for i in sorted(idx_to_pollen)]

if model.feature_names_:
    feature_order = list(model.feature_names_)
else:
    feature_order = [
        "Latitude",
        "Longitude",
        "Elevation",
        "Distance_to_River",
        "Soil",
        "Rainfall",
        "Age",
        "Dominant_Pollen",
    ]

st.title("🏛️ Ancient Human Habitation Predictor")
st.write(
    "Explore geographical, environmental, chronological and pollen "
    "features using this ancient-habitation research prototype."
)

st.warning(
    "Research limitation: the CatBoost model was trained using synthetic "
    "settlement labels. Its output is not archaeological proof, and its "
    "scores are not validated real-world settlement probabilities."
)

tab_home, tab_predict, tab_pollen, tab_model = st.tabs(
    ["Home", "Settlement prediction", "Pollen image classifier", "Model information"]
)

with tab_home:
    st.subheader("Welcome")
    st.write(
        "Use the tabs to explore settlement predictions, classify pollen images, "
        "and inspect model information."
    )
    st.info(
        "This is an exploratory research prototype. Interpret predictions "
        "alongside independent archaeological evidence."
    )

with tab_predict:
    st.subheader("Geographic and environmental inputs")

    last_pollen = st.session_state.get("classified_pollen")
    default_pollen = last_pollen if last_pollen in pollen_names else pollen_names[0]

    with st.form("settlement_form"):
        c1, c2 = st.columns(2)

        with c1:
            lat = st.number_input("Latitude", -90.0, 90.0, 11.0, format="%.6f")
            lon = st.number_input("Longitude", -180.0, 180.0, 79.0, format="%.6f")
            elevation = st.number_input("Elevation (m)", value=10.0)
            river = st.number_input(
                "Distance to nearest river (km)", min_value=0.0, value=2.0
            )

        with c2:
            soil = st.selectbox(
                "Soil type",
                ["Silt", "Rocky", "Loam", "Clay Loam", "Gravel", "Sand"],
            )
            rainfall = st.number_input(
                "Rainfall (mm/year)", min_value=0.0, value=1000.0
            )
            age = st.number_input(
                "Age (years before present)", min_value=0.0, value=2000.0
            )
            pollen = st.selectbox(
                "Dominant pollen class",
                pollen_names,
                index=pollen_names.index(default_pollen),
            )

        submitted = st.form_submit_button(
            "Estimate settlement class", type="primary"
        )

    if last_pollen:
        st.info(f"Latest uploaded-image pollen prediction: {last_pollen}")

    if submitted:
        row = {
            "Latitude": lat,
            "Longitude": lon,
            "Elevation": elevation,
            "Distance_to_River": river,
            "Soil": soil,
            "Rainfall": rainfall,
            "Age": age,
            "Dominant_Pollen": pollen,
        }

        missing_features = [f for f in feature_order if f not in row]

        if missing_features:
            st.error(f"Missing model features: {missing_features}")
        else:
            features = pd.DataFrame(
                [[row[f] for f in feature_order]],
                columns=feature_order,
            )

            try:
                pred = int(np.asarray(model.predict(features)).ravel()[0])
                probs = model.predict_proba(features)[0]
                class_values = list(model.classes_)
                pred_pos = class_values.index(pred)
                confidence = float(probs[pred_pos])

                st.subheader("Prediction result")

                if pred == 1:
                    st.success("Model output: Ancient Human Settlement")
                    st.markdown("## HUMAN SETTLEMENT: YES")
                else:
                    st.info("Model output: No Ancient Human Settlement")
                    st.markdown("## HUMAN SETTLEMENT: NO")

                st.metric("Model score for predicted class", f"{confidence:.1%}")

                prob_df = pd.DataFrame({
                    "Class": [f"Class {c}" for c in class_values],
                    "Model output": probs,
                })

                st.plotly_chart(
                    px.bar(
                        prob_df,
                        x="Class",
                        y="Model output",
                        range_y=[0, 1],
                        title="CatBoost class scores",
                    ),
                    use_container_width=True,
                )

                st.caption(
                    "These scores come from a model trained on synthetic labels "
                    "and are not calibrated historical probabilities."
                )

                st.subheader("Input summary")
                st.dataframe(features, use_container_width=True)

                result = features.copy()
                result["Predicted_Class"] = pred
                result["Predicted_Label"] = (
                    "Ancient Human Settlement"
                    if pred == 1
                    else "No Ancient Human Settlement"
                )
                result["Predicted_Class_Score"] = confidence

                st.download_button(
                    "Download prediction and inputs",
                    result.to_csv(index=False),
                    file_name="habitation_prediction.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Settlement prediction failed: {e}")

with tab_pollen:
    st.subheader("Pollen image classification")
    st.write(
        "Upload a pollen image to classify it. You can use the predicted "
        "class as an input in the settlement prediction tab."
    )

    upload = st.file_uploader(
        "Choose a pollen image",
        type=["jpg", "jpeg", "png"],
        key="pollen_upload",
    )

    if upload is not None:
        from PIL import Image

        try:
            image = Image.open(upload).convert("RGB")
            st.image(image, caption="Uploaded pollen image", width=320)

            resized = image.resize((224, 224))
            arr = np.asarray(resized, dtype=np.float32)[None, ...]

            scores = np.asarray(
                pollen_model.predict(arr, verbose=0)
            )[0]

            pred_idx = int(np.argmax(scores))

            if len(scores) != len(idx_to_pollen) or pred_idx not in idx_to_pollen:
                st.error(
                    "The class mapping does not match the model output. "
                    "Check the original training class order."
                )
            else:
                pred_name = idx_to_pollen[pred_idx]
                st.session_state["classified_pollen"] = pred_name

                st.success(f"Predicted pollen class: {pred_name}")
                st.metric("Model score", f"{float(scores[pred_idx]):.1%}")

                score_df = pd.DataFrame({
                    "Pollen class": [idx_to_pollen[i] for i in range(len(scores))],
                    "Model score": scores,
                }).sort_values("Model score", ascending=False)

                st.plotly_chart(
                    px.bar(
                        score_df.head(10),
                        x="Model score",
                        y="Pollen class",
                        orientation="h",
                        title="Top pollen class scores",
                    ),
                    use_container_width=True,
                )

                st.download_button(
                    "Download pollen classification",
                    pd.DataFrame([{
                        "Predicted_Pollen_Class": pred_name,
                        "Model_Score": float(scores[pred_idx]),
                    }]).to_csv(index=False),
                    file_name="pollen_classification.csv",
                    mime="text/csv",
                )

                st.caption(
                    "The pollen model score is not a calibrated probability. "
                    "Verify the class mapping and preprocessing against the "
                    "original training artifacts."
                )

        except Exception as e:
            st.error(f"Pollen classification failed: {type(e).__name__}: {e}")

with tab_model:
    st.subheader("Model information")
    st.write("Settlement model: CatBoost binary classifier.")
    st.write("Pollen model: MobileNetV2 image classifier.")
    st.write("Settlement input features:")
    st.code("\n".join(feature_order))
    st.write(f"Number of pollen classes: {len(idx_to_pollen)}")
    st.write("Pollen classes:")
    st.write(", ".join(pollen_names))

    try:
        importance = pd.DataFrame({
            "Feature": feature_order,
            "Importance": model.get_feature_importance(),
        }).sort_values("Importance", ascending=True)

        st.plotly_chart(
            px.bar(
                importance,
                x="Importance",
                y="Feature",
                orientation="h",
                title="CatBoost feature importance",
            ),
            use_container_width=True,
        )
    except Exception as e:
        st.info(f"Feature importance is unavailable: {e}")

st.divider()
st.caption(
    "Research prototype. Outputs should be interpreted alongside archaeological "
    "evidence and independent validation, not as proof of ancient habitation."
)