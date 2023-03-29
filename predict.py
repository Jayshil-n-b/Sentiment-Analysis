import pickle
import pandas as pd
import streamlit as st
model = pickle.load(open('model.pkl', 'rb'))
vect = pickle.load(open('vect.pkl', 'rb'))


def predict(rev):
    transform_sent = vect.transform([rev])
    ans = model.predict(transform_sent)
    return ans[0]


def predictText():
    transform_sent = vect.transform([review])
    ans = model.predict(transform_sent)
    return ans[0]


def process(df):
    df['prediction'] = df["review"].apply(predict)
    return df.to_csv().encode("utf-8")


st.title('Sentiment prediction')

review = st.text_input("Enter your review", value="")

if st.button('Predict Sentiment'):
    sent = predictText()
    st.subheader("Your sentiment is: " + sent)

st.markdown("""---""")

review_csv = st.file_uploader("Upload CSV file here", type="csv")

if review_csv is not None:
    df = pd.read_csv(review_csv)
    out_csv = process(df)
    st.download_button(
        label="Download predicted CSV",
        data=out_csv,
        file_name="predictedReviews.csv",
        mime="text/csv",
    )
