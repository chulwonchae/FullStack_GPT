# terminal : source env/bin/activate
# terminal : streamlit run Home.py
# to kill the server : ctl + c

import streamlit as st

with st.sidebar: # 이런식으로 구분 
    st.sidebar.title("sidebar title")
    st.sidebar.text_input('xxx')
    
st.title("title")


tab_one, tab_two, tab_three = st.tabs(['A','B','C'])
with tab_one:
    st.write('test_a')
    
with tab_two:
    st.write('test_b') 

with tab_three:
    st.write('test_c')   