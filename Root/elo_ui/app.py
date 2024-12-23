# ELO_Scoring_and_UI_from_starburt/app.py

import os
import random
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from vectob_db.vectorstore import create_vector_store_from_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from .elo import update_elo
embeddings_list = [
    {"name": "all-MiniLM-L6-v2", "model_name": "sentence-transformers/all-MiniLM-L6-v2"},
    {"name": "all-mpnet-base-v2", "model_name": "sentence-transformers/all-mpnet-base-v2"},
]

chunk_strategies = [
    {"chunk_size": 1000, "chunk_overlap": 200},
    {"chunk_size": 2000, "chunk_overlap": 300},
    {"chunk_size": 500,  "chunk_overlap": 100},
]

configurations = []
for emb in embeddings_list:
    for cs in chunk_strategies:
        configurations.append({
            "embedding_name": emb["name"],
            "embedding_model": emb["model_name"],
            "chunk_size": cs["chunk_size"],
            "chunk_overlap": cs["chunk_overlap"]
        })

def init_session_states():
    if "elo_ratings" not in st.session_state:
        st.session_state.elo_ratings = {i: 1500 for i in range(len(configurations))}
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame(columns=["prompt", "embedding_name", "chunk_size", "chunk_overlap", "elo_rating"])
    if "game_active" not in st.session_state:
        st.session_state.game_active = False
    if "current_config_ids" not in st.session_state:
        st.session_state.current_config_ids = None
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = None
    if "current_answers" not in st.session_state:
        st.session_state.current_answers = {"A": "", "B": ""}

def create_qa_chain(vectorstore, google_api_key: str, model_name="gemini-1.5-pro"):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_tokens=None,
        max_retries=2,
        api_key=google_api_key
    )
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def run_round(pdf_path, google_api_key, prompts):
    config_ids = random.sample(range(len(configurations)), 2)
    configA = configurations[config_ids[0]]
    configB = configurations[config_ids[1]]
    selected_prompt = random.choice(prompts)

    vectorstoreA = create_vector_store_from_pdf(pdf_path, configA["embedding_model"], configA["chunk_size"], configA["chunk_overlap"])
    qa_chainA = create_qa_chain(vectorstoreA, google_api_key)
    answerA = qa_chainA.run(selected_prompt)

    vectorstoreB = create_vector_store_from_pdf(pdf_path, configB["embedding_model"], configB["chunk_size"], configB["chunk_overlap"])
    qa_chainB = create_qa_chain(vectorstoreB, google_api_key)
    answerB = qa_chainB.run(selected_prompt)

    st.session_state.current_config_ids = config_ids
    st.session_state.current_prompt = selected_prompt
    st.session_state.current_answers = {"A": answerA, "B": answerB}

def record_vote(winner):
    config_ids = st.session_state.current_config_ids
    configA = configurations[config_ids[0]]
    configB = configurations[config_ids[1]]

    RA = st.session_state.elo_ratings[config_ids[0]]
    RB = st.session_state.elo_ratings[config_ids[1]]
    RA_new, RB_new = update_elo(RA, RB, winner)
    st.session_state.elo_ratings[config_ids[0]] = RA_new
    st.session_state.elo_ratings[config_ids[1]] = RB_new

    new_row_A = pd.DataFrame([{
        "prompt": st.session_state.current_prompt,
        "embedding_name": configA['embedding_name'],
        "chunk_size": configA['chunk_size'],
        "chunk_overlap": configA['chunk_overlap'],
        "elo_rating": RA_new
    }])
    st.session_state.results_df = pd.concat([st.session_state.results_df, new_row_A], ignore_index=True)

    new_row_B = pd.DataFrame([{
        "prompt": st.session_state.current_prompt,
        "embedding_name": configB['embedding_name'],
        "chunk_size": configB['chunk_size'],
        "chunk_overlap": configB['chunk_overlap'],
        "elo_rating": RB_new
    }])
    st.session_state.results_df = pd.concat([st.session_state.results_df, new_row_B], ignore_index=True)

def run_app():
    st.title("PDF QA Experiment (Randomized, Gamified)")

    init_session_states()

    pdf_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    google_api_key = st.text_input("Google API Key (required)", type="password")
    prompts_text = st.text_area("Enter your prompts (one per line):")
    prompts = [p.strip() for p in prompts_text.split("\n") if p.strip()]

    col_start, col_regress = st.columns([1,1])
    start_game = col_start.button("Start Game")
    run_regression = col_regress.button("Run Regression")

    if start_game:
        if pdf_file is None:
            st.error("No PDF? No party.")
            return
        if not google_api_key:
            st.error("Gotta pay the AI bills. Enter your key.")
            return
        if not prompts:
            st.error("Help me help you—give me at least one prompt.")
            return
        st.session_state.game_active = True

    if st.session_state.game_active:
        pdf_path = "temp_uploaded.pdf"
        if pdf_file is not None:
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.read())

            if st.session_state.current_config_ids is None:
                run_round(pdf_path, google_api_key, prompts)

            st.write("### Answer A:")
            st.write(st.session_state.current_answers["A"])
            st.write("### Answer B:")
            st.write(st.session_state.current_answers["B"])

            colA, colB = st.columns(2)
            if colA.button("A is better"):
                record_vote('A')
                run_round(pdf_path, google_api_key, prompts)
            if colB.button("B is better"):
                record_vote('B')
                run_round(pdf_path, google_api_key, prompts)

    if run_regression:
        df = st.session_state.results_df
        if df.empty:
            st.warning("We need more data. Vote, vote, vote.")
        else:
            st.write("### Regression Analysis")
            X = pd.get_dummies(df[["embedding_name", "chunk_size", "chunk_overlap", "prompt"]], drop_first=True)
            y = df["elo_rating"]

            if len(X) < 2:
                st.warning("Not enough points. Talk to me after you vote more.")
            else:
                model = LinearRegression()
                model.fit(X, y)
                coefficients = pd.Series(model.coef_, index=X.columns)
                st.write("**Coefficients**:")
                st.write(coefficients)
                st.write(f"**Intercept:** {model.intercept_}")
                st.write("Statistics needs more data to be meaningful.")