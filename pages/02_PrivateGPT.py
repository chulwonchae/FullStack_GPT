# Ollma는 내 로컬에 서버 생성
# terminal : source env/bin/activate
# terminal : streamlit run Home.py
# to kill the server : ctl + c
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="📃",
)

#ChatCallbackHandler : llm에서 어떤일이 발생event하면 모든 method들이 호출된다. 
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""  # LLM이 사직되면  empty text생성
    # llm이 언제 토큰 생성 시작하는지 
    def on_llm_start(self, *args, **kwargs): # can have unlimited args and key word arg
        self.message_box = st.empty() # LLM이 사직되면  empty box생성
    # llm이 언제 끝나는지, full message가 되면 
    def on_llm_end(self, *args, **kwargs): 
        save_message(self.message, "ai")
    # 토큰을 message에 합쳐
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token 
        self.message_box.markdown(self.message) # 메시지를 방금 업데이트 한 메세지와 markdown text로 제공


llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(), # 기본적으로 llm의 event를 listen하는 class
    ],
)

# 이 computation을 cache한다. 파일이 변경되지 않는 한, 동일한 파일을 함수에 계속 보내면서 함수는 작동되지 않음
@st.cache_resource(show_spinner="Embedding file...") 
def embed_file(file):  
    file_content = file.read() # 파일 읽고
    file_path = f"./.cache/private_files/{file.name}" # 저장하고
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")#저장하고
    splitter = CharacterTextSplitter.from_tiktoken_encoder( #split하고
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path) # load
    docs = loader.load_and_split(text_splitter=splitter) # doc split
    embeddings = OllamaEmbeddings(model='mistral:latest') #embed
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir) # cached된 거에서 임베딩을 가져오고
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()#벡터스토어를 리트리버로 바꾸고 
    return retriever # 체인으로 간다; 결국 그냥 다큐먼트를 주려는게 목적 단순


def save_message(message, role): # 딕셔너리를 더하는  gkatn
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role) # 메세지 저장


def paint_history(): #message store의 모든 메세지에 대해서 이 send_message를 통해 화면에 출력
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False, # store에 저장하지 않고(이미 저장된것들이니까)
        )


def format_docs(docs): # string으로 doc을 줄것이야 (linebreak 해서)
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using ONLY the following context and not your training data. If you don't know the answer just say you don't know. DON'T make anything up.
        Context: {context} 
        Question: {question}
        """ 
        )



st.title("PrivateGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)
# 1_1. File upload in the side bar
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )
# 1_3. user가 파일을 업로드 하면 우리는 파일로부터 retriever얻음
if file:
    retriever = embed_file(file) # embed_file 함수 가동 
    send_message("I'm ready! Ask away!", "ai", save=False) # send_message 가동
    paint_history() #paint_history작동 
    message = st.chat_input("Ask anything about your file...") # 사용자가 질문할 chat input생성
    if message: # 
        send_message(message, "human") #메세지를 화면에 표시하고 저장
        chain = (
            { 
                "context": retriever | RunnableLambda(format_docs), #'retriever'에서 list of doc들 줄것이고 --> format_docs
                "question": RunnablePassthrough(), #invoke 하면 messages (사용자 질문)
            }
            | prompt 
            | llm
        )
        with st.chat_message("ai"): #대답을 에이아이 아이콘 옆에서
            response = chain.invoke(message)

# 1_2. 처음에는 파일이 없을 것임, 그러면 session state에 있는 메세지 저장소를 empty list 로 초기화
else:
    st.session_state["messages"] = []