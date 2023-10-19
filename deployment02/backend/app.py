#! /usr/bin/env python3
# coding: utf-8

import os
from flask import Flask, request, redirect, jsonify, url_for, session, abort
from flask_cors import CORS

from llm_setup import init_llm_retriever, execute_code

# ########## INIT APP ##########

# --- API Flask app ---
app = Flask(__name__)
app.secret_key = "super secret key"
# app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 200

CORS(app)


# ########## Init LLM retriever ##########

llm_retriever = init_llm_retriever("SaloniJhalani/ft-falcon-7b-instruct")
# llm_retriever = init_llm_retriever("TheBloke/CodeLlama-7B-Python-GPTQ")
# llm_retriever = init_llm_retriever("TheBloke/CodeLlama-13B-Python-GPTQ")
# llm_retriever = init_llm_retriever("TheBloke/Mistral-7B-Instruct-v0.1-GPTQ")

# ########## API ENTRY POINTS (BACKEND) ##########

@app.route("/")
# @app.doc(hide=True)
def index():
    """Define the content of the main fontend page of the API server"""

    return f"""
    <h1>The 'Omdena's LLM Inference API' server is up.</h1>
    """

@app.route("/inference", methods=['POST'])
def inference():
    """Infer using the LLM model and the retriever"""

    query = request.form.get("query")

    print("\n", f" Infering with the query: \"{query}\"", "\n")

    prompt = f'''You are an experienced Python programmer skilled in creating data visualizations using Matplotlib. You have access to a dataset stored in a CSV file, './library_data.csv,' which contains extensive information about the number of books available for various 'title_en' categories in UAE libraries over multiple years from 2014 to 2016. This dataset includes columns for 'year,' 'title_en,' and 'value,' where 'value' represents the number of books available for a specific 'title_en' in UAE libraries during that year.

    Ensure the dataset to include only the relevant data as per the user query.

    Please ensure that your code is well-documented, including axis labels, titles, and any other necessary components for a clear and informative visualization.

    Return plain code for execution that can be directly executed without any modifications.

    ### Question: {query}
    ### Answer:
    '''

    result = execute_code(llm_retriever.run(prompt))
    print(result)

    print(jsonify(result))
    return jsonify(result)


# ########## START FLASK SERVER ##########

if __name__ == "__main__":

    current_port = int(os.environ.get("PORT") or 5000)
    app.run(debug=False, host="0.0.0.0", port=current_port, threaded=True)
