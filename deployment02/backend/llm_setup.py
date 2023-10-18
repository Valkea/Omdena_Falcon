
### ===== Load libraries =====

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

from huggingface_hub import login as hf_login
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel


import torch
from torch import cuda

import locale
locale.getpreferredencoding = lambda: "UTF-8"


def prepare_data():

    # ----- Data Parsing

    library = CSVLoader("library_data.csv")
    library_data = library.load()
    # library_data[0]

    # ----- Text Splitter

    text_splitter = TokenTextSplitter(
        chunk_size=1000,
        chunk_overlap = 200,
    )

    library_doc = text_splitter.split_documents(library_data)
    # library_doc[0]

    return library_doc


def prepare_data_retriever(library_doc):

    # ----- Index / Vector Store (FAISS)

    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    core_embeddings_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )

    # CacheBackedEmbeddings saves time and money when user asks same question.
    store = LocalFileStore("./cache/")
    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model, store, namespace=embed_model_id
    )

    vector_store = FAISS.from_documents(library_doc, embedder)

    # ----- Check if the vectorstore is working correctly.
    #
    # query = "In python, write a code that reads the csv file and plot a scatter plot of x-axis labeled 'Year' and the y-axis labeled 'value'"
    # 
    # embedding_vector = core_embeddings_model.embed_query(query)
    # docs = vector_store.similarity_search_by_vector(embedding_vector, k=3)
    # 
    # for page in docs:
    #   print(page.page_content)

    # ----- Build retriever
    # 
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    # docs = retriever.get_relevant_documents("In python, write a code that reads the csv file and plot a scatter plot of x-axis labeled 'Year' and the y-axis labeled 'value'")

    return retriever


def load_llm(model_id):

    hf_login(token="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") # ENV

    # ----- Load model directly

    if model_id == "SaloniJhalani/ft-falcon-7b-instruct":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] == 8 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype = dtype, #torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id,device_map='cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        return_full_text=True,
        temperature=0.0,
        max_new_tokens=1024,        # a higher number of tokens delays the prompt
        repetition_penalty=1.1      # avoid repeating
    )

    # result = generate_text("Write a code that plot a bar graph to display the value of 'Philosophy and psychology' title_en over the years?")
    # result[0]["generated_text"]

    llm = HuggingFacePipeline(pipeline=generate_text)
    return llm


def prepare_llm(llm, retriever):

    # ----- Template for an instruction with no input

    prompt = PromptTemplate(
        input_variables=["instruction"],
        template="{instruction}"
    )

    # ----- LLMChain
    #
    # llm_chain = LLMChain(llm=llm, prompt=prompt)
    # 
    # print(llm_chain.predict(
    #     instruction="Write a code that plot a bar graph to display the value of 'Philosophy and psychology' title_en over the years?"
    # ).lstrip())

    # ----- RetrievalQA

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
        )

    return qa



def execute_code(code):
    """ Parse and execute the returned python code """

    # Remove "```python" at the beginning
    code = code.replace("```python", "")

    # Remove "```" at the end
    code = code.replace("```", "")
    code = code.replace('"""', "")
    code = code.split("###")[0]

    try:
      exec(code)
    except Exception as e:
      print(f"Error executing code:{str(e)}")
    return code


def init_llm_retriever(model_id):

    print("\n", " Initialize the chat components ".center(100, "*"), "\n")

    library_doc = prepare_data()
    retriever = prepare_data_retriever(library_doc)
    llm = load_llm(model_id)
    qa = prepare_llm(llm, retriever)

    print("\n", " LLM is ready ".center(100, "*"), "\n")

    return qa


if __name__ == "__main__":
    qa = init_llm_retriever("TheBloke/CodeLlama-7B-Python-GPTQ")
