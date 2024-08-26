import os
import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import openai
import glob
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable.passthrough import RunnablePassthrough
from langchain.schema.runnable.base import RunnableLambda
from langchain.storage import LocalFileStore


llm = ChatOpenAI(temperature=0.1, streaming=True)

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)


has_transcript = os.path.exists("./.cache/videosample.txt") # 이미 있는 파일

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = glob.glob(f"{chunk_folder}/*.mp3") 
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript["text"])


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]

    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len

        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.
Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    video_path = f"./.cache/{video.name}"
    audio_path = video_path.replace("mp4", "mp3")
    chunks_folder = "./.cache/chunks"
    transcript_path = video_path.replace("mp4", "txt")
    with st.status("Loading video...") as status:  #status화면
        with open(video_path, "wb") as file:
            file.write(video.read())
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        ["Transcript", "Summary", "Q&A"],
    )

    with transcript_tab: # transcript 화면 
        with open(transcript_path, "r") as file:
            st.write(file.read())
    #summary "REFINE" - doc하나씩 summary
    with summary_tab: # 
        start = st.button("Generate Summary")

        if start:
            loader = TextLoader(transcript_path)
            docs = loader.load_and_split(text_splitter=splitter)
            first_summary_prompt = ChatPromptTemplate.from_template( # 처음 doc summary
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:                
            """
            )

            first_summary_chain = first_summary_prompt | llm | StrOutputParser() # string

            summary = first_summary_chain.invoke(
                {
                    "text": docs[0].page_content,
                }
            )

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing...") as status: # summary gets refine...
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1} ")
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )
                    st.write(summary)
            st.write(summary)

    with qa_tab:
        retriever = embed_file(transcript_path)

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
                ),
                ("human", "{question}"),
            ]
        )

        message = st.text_input("Ask anything about your video...")

        if message:
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | qa_prompt
                | llm
            )

            with st.chat_message("ai"):
                chain.invoke(message).content