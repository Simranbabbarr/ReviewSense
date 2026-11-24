
import os
import pandas as pd
import streamlit as st
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "simranbabbarr/reviewsense-distilbert"
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
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0;
}
.subtitle {
    color: #8b929d;
    margin-top: .3rem;
    margin-bottom: 1.5rem;
}
.result-label {
    font-size: 1.7rem;
    font-weight: 750;
    text-transform: uppercase;
}
.small-note {
    font-size: .85rem;
    color: #8b929d;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading ReviewSense AI model...")
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
    if not Path(DATA_PATH).exists():
        return None

    for encoding in ("cp1252", "utf-8", "latin1"):
        try:
            return pd.read_csv(DATA_PATH, encoding=encoding)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(DATA_PATH)


try:
    tokenizer, model = load_model()
    labels = get_labels(model)

except Exception as error:
    st.error(f"Could not load the ReviewSense model: {error}")
    st.info("Check your internet connection and Hugging Face model repository.")
    st.stop()


dataset = read_dataset()

st.markdown(
    '<p class="main-title">ReviewSense 📝</p>',
    unsafe_allow_html=True
)

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

    st.subheader("Analyse an Individual Review")

    left, right = st.columns([1.4, 1])

    with left:

        examples = [
            "The professor explains every concept clearly and makes learning interesting.",
            "The classes are confusing and the professor rarely answers questions.",
            "Teaching is acceptable, but the lectures could be more engaging.",
            "The infrastructure and facilities are excellent.",
            "The overall academic experience is disappointing."
        ]

        selected_example = st.selectbox(
            "Try an example review",
            ["Write my own review"] + examples
        )

        default_text = (
            ""
            if selected_example == "Write my own review"
            else selected_example
        )

        review = st.text_area(
            "Student Review",
            value=default_text,
            height=190,
            placeholder="Type or paste a student review here..."
        )

        predict_button = st.button(
            "Analyse Review",
            type="primary",
            use_container_width=True
        )

    with right:

        st.markdown("#### What ReviewSense provides")

        st.write("✓ Five-class review classification")
        st.write("✓ Model confidence score")
        st.write("✓ Category-wise probabilities")
        st.write("✓ Review word count")
        st.write("✓ Transformer-based predictions")

        st.caption(
            "Predictions reflect patterns learned from the training dataset "
            "and should be interpreted as model estimates."
        )

    if predict_button:

        if not review.strip():
            st.warning("Please enter a review before analysing.")

        else:
            with st.spinner("Analysing your review..."):
                result = predict_texts(
                    [review],
                    tokenizer,
                    model
                )[0]

                st.session_state["last_review"] = review
                st.session_state["last_result"] = result

    if "last_result" in st.session_state:

        result = st.session_state["last_result"]
        review_text = st.session_state["last_review"]

        st.divider()
        st.subheader("Prediction Result")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Predicted Category",
            result["prediction"].upper()
        )

        metric2.metric(
            "Model Confidence",
            f'{result["confidence"]:.2f}%'
        )

        metric3.metric(
            "Word Count",
            len(review_text.split())
        )

        st.markdown("#### Submitted Review")
        st.info(review_text)

        probability_frame = pd.DataFrame({
            "Category": list(result["probabilities"].keys()),
            "Probability (%)": list(result["probabilities"].values())
        }).set_index("Category")

        st.markdown("#### Category Probability Distribution")

        st.bar_chart(
            probability_frame,
            horizontal=True
        )

        with st.expander("View detailed probability values"):
            st.dataframe(
                probability_frame,
                use_container_width=True
            )


with tab_batch:

    st.subheader("Batch Review Analysis")

    st.write(
        "Upload a CSV file containing multiple student reviews "
        "to classify them together."
    )

    uploaded_file = st.file_uploader(
        "Upload Review CSV",
        type=["csv"],
        key="batch_upload"
    )

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)

            st.markdown("#### Dataset Preview")

            st.dataframe(
                batch_df.head(10),
                use_container_width=True
            )

            text_columns = batch_df.select_dtypes(
                include=["object", "string"]
            ).columns.tolist()

            if not text_columns:

                st.error("No text columns were found in this CSV.")

            else:

                default_index = (
                    text_columns.index("comment")
                    if "comment" in text_columns
                    else 0
                )

                text_column = st.selectbox(
                    "Select Review Text Column",
                    text_columns,
                    index=default_index
                )

                if st.button(
                    "Classify All Reviews",
                    type="primary"
                ):

                    texts = (
                        batch_df[text_column]
                        .fillna("")
                        .astype(str)
                        .tolist()
                    )

                    valid_indices = [
                        index
                        for index, value in enumerate(texts)
                        if value.strip()
                    ]

                    valid_texts = [
                        texts[index]
                        for index in valid_indices
                    ]

                    if not valid_texts:

                        st.warning(
                            "The selected column contains no valid reviews."
                        )

                    else:

                        with st.spinner(
                            f"Classifying {len(valid_texts)} reviews..."
                        ):

                            predictions = predict_texts(
                                valid_texts,
                                tokenizer,
                                model
                            )

                        output = batch_df.copy()

                        output["predicted_quality"] = ""
                        output["confidence_percent"] = None

                        for original_index, prediction in zip(
                            valid_indices,
                            predictions
                        ):

                            output.at[
                                original_index,
                                "predicted_quality"
                            ] = prediction["prediction"]

                            output.at[
                                original_index,
                                "confidence_percent"
                            ] = prediction["confidence"]

                        st.session_state["batch_output"] = output

                        st.success(
                            f"Successfully classified {len(valid_texts)} reviews!"
                        )

            if "batch_output" in st.session_state:

                output = st.session_state["batch_output"]

                st.markdown("#### Classification Results")

                st.dataframe(
                    output,
                    use_container_width=True
                )

                st.download_button(
                    "Download Classified CSV",
                    data=output.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="reviewsense_predictions.csv",
                    mime="text/csv"
                )

        except Exception as error:

            st.error(f"Could not process CSV: {error}")


with tab_insights:

    st.subheader("Dataset Insights")

    if dataset is None:

        st.info(
            "Upload your original dataset into the project folder "
            "as feedback student.csv to enable dataset insights."
        )

    elif not {"comment", "quality"}.issubset(dataset.columns):

        st.warning(
            "The dataset must contain comment and quality columns."
        )

    else:

        clean_data = dataset[
            ["comment", "quality"]
        ].dropna().copy()

        clean_data["comment"] = (
            clean_data["comment"]
            .astype(str)
            .str.strip()
        )

        clean_data["quality"] = (
            clean_data["quality"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        clean_data = clean_data[
            clean_data["comment"] != ""
        ]

        total_reviews = len(clean_data)
        total_categories = clean_data["quality"].nunique()

        average_words = (
            clean_data["comment"]
            .apply(lambda value: len(value.split()))
            .mean()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Reviews",
            f"{total_reviews:,}"
        )

        c2.metric(
            "Review Categories",
            total_categories
        )

        c3.metric(
            "Average Words",
            f"{average_words:.1f}"
        )

        st.markdown("#### Review Distribution")

        counts = (
            clean_data["quality"]
            .value_counts()
            .rename_axis("Category")
            .to_frame("Reviews")
        )

        st.bar_chart(
            counts,
            horizontal=True
        )

        st.markdown("#### Filter Reviews by Category")

        category_filter = st.selectbox(
            "Choose Category",
            ["All"] + sorted(
                clean_data["quality"].unique().tolist()
            )
        )

        samples = (
            clean_data
            if category_filter == "All"
            else clean_data[
                clean_data["quality"] == category_filter
            ]
        )

        st.dataframe(
            samples.head(50),
            use_container_width=True
        )


with tab_about:

    st.subheader("About ReviewSense")

    st.markdown("""
    ReviewSense is a transformer-based student review
    classification application designed to analyse feedback
    and identify review quality categories.
    """)

    st.markdown("#### Model Architecture")

    st.write("**Architecture:** DistilBERT")
    st.write("**Model Type:** Transformer")
    st.write("**Task:** Multi-class Text Classification")
    st.write("**Maximum Token Length:** 256")
    st.write(
        "**Categories:** "
        + ", ".join(
            labels[index]
            for index in sorted(labels)
        )
    )

    st.markdown("#### Model Repository")

    st.link_button(
        "View Model on Hugging Face",
        "https://huggingface.co/simranbabbarr/reviewsense-distilbert"
    )

    st.markdown("#### How It Works")

    st.write(
        "1. Student review is entered or uploaded."
    )

    st.write(
        "2. The tokenizer converts text into model inputs."
    )

    st.write(
        "3. Fine-tuned DistilBERT processes the text."
    )

    st.write(
        "4. The classification layer generates category probabilities."
    )

    st.write(
        "5. ReviewSense displays the predicted category and confidence."
    )

    st.markdown("#### Responsible Use")

    st.info(
        "Predictions are based on patterns learned from the supplied "
        "training dataset. They should be used as an initial feedback "
        "analysis tool, not as the sole basis for decisions about "
        "faculty or academic institutions."
    )


st.divider()

st.caption(
    "ReviewSense | Transformer-Based Student Review Classification | "
    "Powered by Fine-Tuned DistilBERT"
)
