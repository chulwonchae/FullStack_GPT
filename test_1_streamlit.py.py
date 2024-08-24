# terminal : source env/bin/activate
# terminal : streamlit run Home.py
# to kill the server : ctl + c

import streamlit as st
from langchain.prompts import PromptTemplate
from datetime import datetime
# st.title("Hello WORLD!")
# st.subheader("Welcome to Streamlit!")
# st.markdown("""
#             #### I love it!
#             """)
# st.write("hello")
# st.write([1,2,3,4])
# st.write({'x':1 })
# st.write(PromptTemplate)
#p = PromptTemplate.from_template("xxxx")
#st.write(p)
#st.selectbox("Choose your model", ("GPT-3", "GPT-4"))


# today = datetime.today().strftime("%H:%M:%S")
# st.title(today)
# model = st.selectbox(
#     "Choose your model",
#     (
#         "GPT-3",
#         "GPT-4",
#     ),
# )

# if model == "GPT-3":
#     st.write("cheap")
# else:
#     st.write("not cheap")
# name = st.text_input("What is your name?")
# st.write(name)

# value = st.slider(
#     "temperature",
#     min_value=0.1,
#     max_value=1.0,
#     )

# st.write(value)