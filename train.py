import joblib
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from constants import DATA_FILE, MODEL_DIR, MODEL_FILE, TARGET, TEST_SIZE, RANDOM_STATE


def train_and_save_model():
    try:
        logger.info("Training Pipeline started")
        # Perform data ingestion
        logger.info("Data ingestion started")
        df = pd.read_csv(DATA_FILE)
        logger.info(f"Data successfully loaded with shape : {df.shape}")
        logger.info(f"First 3 rows of dataframe :\n{df.head(3)}")
        # Check for duplicates in dataframe
        logger.info(f"Duplicate rows : {df.duplicated().sum()}")
        # Drop the duplictes
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        logger.info(f"Duplicates dropped shape after dropping duplicated : {df.shape}")
        # Check for missing values
        logger.info(f"Missing values :\n{df.isna().sum()}")
        # Seperate X and Y
        logger.info("Seperating X and Y")
        X = df.drop(columns=[TARGET])
        Y = df[TARGET]
        logger.info(
            f"Seperation of X and Y complete with shapes {X.shape} and {Y.shape}"
        )
        # Apply Train test split o
        logger.info("Applying Train Test split")
        xtrain, xtest, ytrain, ytest = train_test_split(
            X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        # Print shape of
        logger.info(f"Train test split completed")
        logger.info(f"xtrain shape : {xtrain.shape}, ytrain shape : {ytrain.shape}")
        logger.info(f"xtest shape : {xtest.shape}, ytest shape : {ytest.shape}")
        # Create a pipeline with model
        logger.info("Create a model pipeline")
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(random_state=RANDOM_STATE),
        )
        logger.info("Model pipeline created sucessfully")
        logger.info("Calculating 5 fold Cross validated F1 macro")
        scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring="f1_macro")
        logger.info(f"Scores calculated : {scores}")
        logger.info(f"Average score : {scores.mean().round(4)}")
        logger.info(f"Score std dev : {scores.std().round(4)}")
        # Train the model
        logger.info("Training model")
        model.fit(xtrain, ytrain)
        # Evalute f1 macro for the train and test
        logger.info("Evaluating the model")
        ypred_train = model.predict(xtrain)
        ypred_test = model.predict(xtest)
        f1_train = f1_score(ytrain, ypred_train, average="macro")
        f1_test = f1_score(ytest, ypred_test, average="macro")
        gen_err = abs(f1_train - f1_test)
        logger.info(f"F1 macro Train : {f1_train:.4f}")
        logger.info(f"F1 macro Test : {f1_test:.4f}")
        logger.info(f"Generalization error : {gen_err:.4f}")
        logger.success("Model training and evaluation successful")
        # Save the model object
        logger.info(f"Saving the model to : {MODEL_FILE}")
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_FILE)
        logger.success(f"Model successfully saved in {MODEL_FILE}")
    except Exception as e:
        logger.error(f"Exception occured : {e}")


if __name__ == "__main__":
    train_and_save_model()
