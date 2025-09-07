import streamlit as st
from inference import load_model, get_preds_and_probs

# intialize streamlit app
st.set_page_config(page_title="End to End ML Project")

# load the model object
model = load_model()

# Add the title to webpage
st.title("Iris end to end ML Project")
st.subheader("by Utkarsh Gaikwad")

# Take the inputs from user
sep_len = st.number_input("Sepal Length :", min_value=0.00, step=0.01)
sep_wid = st.number_input("Sepal Width :", min_value=0.00, step=0.01)
pet_len = st.number_input("Petal Length :", min_value=0.00, step=0.01)
pet_wid = st.number_input("Petal Width :", min_value=0.00, step=0.01)

# Create a button to predict results
button = st.button("Predict", type="primary")

# If button is clicked
if button:
    pred, prob_df = get_preds_and_probs(model, sep_len, sep_wid, pet_len, pet_wid)
    st.subheader(f"Predicted Species : {pred}")
    st.subheader("Probablility : ")
    st.dataframe(prob_df)
    st.bar_chart(prob_df.T)
