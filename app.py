import os
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "reviewsense_model"
DATA_PATH = "feedback student.csv"
MAX_LENGTH = 256
BATCH_SIZE = 16

st.set_page_config(
    page_title="ReviewSense",
    page_icon="📝",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.main-title {font-size: 2.35rem; font-weight: 750; margin-bottom: 0;}
.subtitle {color: #8b929d; margin-top: .25rem; margin-bottom: 1.4rem;}
.result-label {font-size: 1.7rem; font-weight: 750; text-transform: uppercase;}
.small-note {font-size: .85rem; color: #8b929d;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

def get_labels(model):
    raw = model.config.id2label
    return {int(key): value for key, value in raw.items()}

def predict_texts(texts, tokenizer, model):
    results = []
    labels = get_labels(model)

    for start in range(0, len(texts), BATCH_SIZE):
        batch = [str(text) for text in texts[start:start + BATCH_SIZE]]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=1)

        for row in probabilities:
            values = row.tolist()
            predicted_id = int(torch.argmax(row).item())
            results.append({
                "prediction": labels[predicted_id],
                "confidence": round(values[predicted_id] * 100, 2),
                "probabilities": {
                    labels[index]: round(value * 100, 2)
                    for index, value in enumerate(values)
                }
            })

    return results

def read_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    for encoding in ("cp1252", "utf-8", "latin1"):
        try:
            return pd.read_csv(DATA_PATH, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(DATA_PATH)

if not os.path.isdir(MODEL_PATH):
    st.error("Trained model not found. Run `python train.py` first.")
    st.stop()

try:
    tokenizer, model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

labels = get_labels(model)
dataset = read_dataset()

st.markdown('<p class="main-title">ReviewSense</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Transformer-powered student review analysis using fine-tuned DistilBERT.</p>',
    unsafe_allow_html=True
)

tab_predict, tab_batch, tab_insights, tab_about = st.tabs([
    "Review Prediction",
    "Batch Analysis",
    "Dataset Insights",
    "About Model"
])

with tab_predict:
    st.subheader("Analyse an individual review")
    left, right = st.columns([1.35, 1])

    with left:
        examples = [
            "The professor explains every concept clearly and makes learning interesting.",
            "The classes are confusing and the professor rarely answers questions.",
            "Teaching is acceptable, but the lectures could be more engaging."
        ]
        selected_example = st.selectbox(
            "Try an example review",
            ["Write my own review"] + examples
        )
        default_text = "" if selected_example == "Write my own review" else selected_example
        review = st.text_area(
            "Student review",
            value=default_text,
            height=190,
            placeholder="Type or paste a student review here..."
        )
        predict_button = st.button("Analyse Review", type="primary", use_container_width=True)

    with right:
        st.markdown("#### What this tool provides")
        st.write("• Predicted review category")
        st.write("• Model confidence score")
        st.write("• Probability for every category")
        st.write("• Review length and quick text statistics")
        st.caption("Predictions reflect patterns learned from the supplied dataset; they are not objective evaluations of a professor.")

    if predict_button:
        if not review.strip():
            st.warning("Please enter a review before analysing.")
        else:
            result = predict_texts([review], tokenizer, model)[0]
            st.session_state["last_review"] = review
            st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        review_text = st.session_state["last_review"]
        st.divider()
        st.subheader("Prediction result")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Predicted category", result["prediction"].upper())
        metric2.metric("Model confidence", f'{result["confidence"]:.2f}%')
        metric3.metric("Word count", len(review_text.split()))

        probability_frame = pd.DataFrame({
            "Category": list(result["probabilities"].keys()),
            "Probability (%)": list(result["probabilities"].values())
        }).set_index("Category")
        st.markdown("#### Probability by category")
        st.bar_chart(probability_frame, horizontal=True)
        with st.expander("View probability values"):
            st.dataframe(probability_frame, use_container_width=True)

with tab_batch:
    st.subheader("Classify multiple reviews")
    st.write("Upload a CSV containing a column with student review text.")
    uploaded_file = st.file_uploader("Upload review CSV", type=["csv"], key="batch_upload")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data")
            st.dataframe(batch_df.head(10), use_container_width=True)

            text_columns = batch_df.select_dtypes(include=["object", "string"]).columns.tolist()
            if not text_columns:
                st.error("No text columns were found in this CSV.")
            else:
                default_index = text_columns.index("comment") if "comment" in text_columns else 0
                text_column = st.selectbox(
                    "Select the review text column",
                    text_columns,
                    index=default_index
                )

                if st.button("Classify all reviews", type="primary"):
                    texts = batch_df[text_column].fillna("").astype(str).tolist()
                    valid_indices = [index for index, value in enumerate(texts) if value.strip()]
                    valid_texts = [texts[index] for index in valid_indices]

                    if not valid_texts:
                        st.warning("The selected column does not contain any non-empty reviews.")
                    else:
                        with st.spinner(f"Classifying {len(valid_texts)} reviews..."):
                            predictions = predict_texts(valid_texts, tokenizer, model)

                        output = batch_df.copy()
                        output["predicted_quality"] = ""
                        output["confidence_percent"] = None

                        for original_index, prediction in zip(valid_indices, predictions):
                            output.at[original_index, "predicted_quality"] = prediction["prediction"]
                            output.at[original_index, "confidence_percent"] = prediction["confidence"]

                        st.session_state["batch_output"] = output
                        st.success(f"Finished classifying {len(valid_texts)} reviews.")

            if "batch_output" in st.session_state:
                output = st.session_state["batch_output"]
                st.dataframe(output, use_container_width=True)
                csv_data = output.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Download classified CSV",
                    data=csv_data,
                    file_name="reviewsense_predictions.csv",
                    mime="text/csv"
                )
        except Exception as error:
            st.error(f"Could not process this CSV: {error}")

with tab_insights:
    st.subheader("Dataset overview")
    if dataset is None:
        st.info(f"Place `{DATA_PATH}` in the project folder to view dataset insights.")
    elif not {"comment", "quality"}.issubset(dataset.columns):
        st.warning("The dataset must contain `comment` and `quality` columns for these insights.")
    else:
        clean_data = dataset[["comment", "quality"]].dropna().copy()
        clean_data["comment"] = clean_data["comment"].astype(str).str.strip()
        clean_data["quality"] = clean_data["quality"].astype(str).str.lower().str.strip()
        clean_data = clean_data[clean_data["comment"] != ""]

        total_reviews = len(clean_data)
        total_categories = clean_data["quality"].nunique()
        average_words = clean_data["comment"].apply(lambda value: len(value.split())).mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Usable reviews", f"{total_reviews:,}")
        c2.metric("Review categories", total_categories)
        c3.metric("Average words per review", f"{average_words:.1f}")

        st.markdown("#### Reviews by category")
        counts = clean_data["quality"].value_counts().rename_axis("Category").to_frame("Reviews")
        st.bar_chart(counts, horizontal=True)

        st.markdown("#### Sample reviews")
        category_filter = st.selectbox(
            "Filter by category",
            ["All"] + sorted(clean_data["quality"].unique().tolist())
        )
        samples = clean_data if category_filter == "All" else clean_data[clean_data["quality"] == category_filter]
        st.dataframe(samples.head(20), use_container_width=True)

with tab_about:
    st.subheader("Model information")
    st.write("**Architecture:** DistilBERT")
    st.write("**Task:** Five-class text classification")
    st.write("**Maximum token length:** 256")
    st.write("**Output categories:** " + ", ".join(labels[index] for index in sorted(labels)))
    st.write("**Model directory:** `reviewsense_model/`")
    st.info(
        "The model predicts the quality label patterns present in the training dataset. "
        "Its predictions can be affected by class imbalance, ambiguous reviews, spelling, "
        "sarcasm, and the quality of the original labels."
    )
    st.markdown("#### Responsible use")
    st.write(
        "Use predictions as a first-pass summary of student feedback, not as the sole basis "
        "for decisions about faculty members. Review the original comments before taking action."
    )

st.divider()
st.caption("ReviewSense • Transformer-Based Student Review Classification")
