# terminal : source env/bin/activate
# terminal : streamlit run Home.py
# to kill the server : ctl + c

import streamlit as st
import time

st.set_page_config(
    page_title= "DocumentGPT",
    page_icon="📑"
)
st.title("DocumentGPT")

#session_state : 여러번의 재실행에도 데이터가 보존 
if "messages" not in st.session_state:  # 'messages'가 없다면, 만들어라, 
    st.session_state["messages"] = [] #처음에만 it will be empty list with "messages"


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


# repainting all the saved message(past message)
for message in st.session_state["messages"]: # 만약 session_state안에  message가 있다면
    send_message(      #send_message function 
        message["message"],
        message["role"],
        save=False,
    )


message = st.chat_input("Send a message to the ai ")

# new message
if message:
    send_message(message, "human") # 사람이 message 말하면
    time.sleep(2)
    send_message(f"You said: {message}", "ai") # 메세지 따라하고

    with st.sidebar:
        st.write(st.session_state)



# with st.chat_message("human"): # 사람이 쓰는것
#     st.write('Hellooooo')

# with st.chat_message("ai"):  # ai가 쓰는ㅁ것 
#     st.write('How are you?')

# st.chat_input("Send a message to the ai") # AI에게 메세지 쓰기


# with st.status("Embedding file...", expanded=True) as status: # 모래시계처럼 현상황 돌아가는것 그리고 마지막에 체크
#     time.sleep(2)
#     st.write("Getting the file")
#     time.sleep(2)
#     st.write("Embedding the file")
#     time.sleep(2)
#     st.write("Caching the file")
#     time.sleep(2)
#     status.update(label='Error', state='error')  #에러
