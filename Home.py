# terminal : source env/bin/activate
# terminal : streamlit run Home.py
# to kill the server : ctl + c

import streamlit as st

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🧸"
)

st.title("FullstackGPT Home")

st.markdown(
    """
# Hello! 

Welcome to Chulwon's FullstackGPT Portfolio!

Here are the apps I made :
- [ ] [DocumentGPT](/DocumentGPT)
- [ ] [PritaveGPT](/PritaveGPT)  
- [ ] [QuizGPT](/QuizGPT) 
- [ ] [SiteGPT](/SiteGPT) 
- [ ] [MeetingGPT](/MeetingGPT) 
- [ ] [InvestorGPT](/InvestorGPT) 
    """
)