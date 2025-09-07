import joblib
import pandas as pd
from loguru import logger
from pathlib import Path
from constants import MODEL_FILE
from sklearn.pipeline import Pipeline
import streamlit as st


@st.cache_resource()
def load_model(file_path: Path = MODEL_FILE):
    logger.info(f"Loading model from : {file_path}")
    model = joblib.load(file_path)
    logger.info(f"Model object loaded successfully")
    return model


def get_preds_and_probs(
    model: Pipeline,
    sep_len: float | int,
    sep_wid: float | int,
    pet_len: float | int,
    pet_wid: float | int,
) -> tuple:
    try:
        # Predict the results
        logger.info("Getting data as dataframe")
        data = [
            {
                "sepal_length": sep_len,
                "sepal_width": sep_wid,
                "petal_length": pet_len,
                "petal_width": pet_wid,
            }
        ]
        xnew = pd.DataFrame(data)
        logger.info(f"Data loaded successfully as dataframe :\n{xnew}")
        # Predict the result
        logger.info("Predicting results")
        pred = model.predict(xnew)[0]
        logger.info(f"Predicted species : {pred}")
        # Predict the probability
        logger.info("Predicting probabilites")
        prob = model.predict_proba(xnew).round(4)
        prob_df = pd.DataFrame(prob, columns = model.classes_)
        logger.info(f"Probabilites calculated successfully :\n{prob_df}")
        return pred, prob_df
    except Exception as e:
        logger.error(f"Error occured during inference : {e}")


if __name__ == "__main__":
    sep_len = 5.9
    sep_wid = 3.0
    pet_len = 4.2
    pet_wid = 1.5
    model = load_model()
    pred, prob_df = get_preds_and_probs(model, sep_len, sep_wid, pet_len, pet_wid)
