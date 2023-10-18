import gradio as gr
import requests

base_url = "0.0.0.0:5000"

def predict(url, query):
    data = {"query": query}

    print("Infering...", data)
    res=requests.post(f"{url}/inference", data=data)
    print(res.json())
    return res.json()

examples = [
    ["Create a Python code to generate a pie chart that visualizes the distribution of 'title_en' categories for the year 2014."],
    ["Your task is to generate Python code to create a multiple line graph that illustrates the trends for the 'Linguistics' and 'Philosophy and psychology' title_en over the years. Please utilize Matplotlib for this data visualization."],
    ["Generate Python code to create multiple bar graphs showing the availability trends of 'Philosophy and psychology' and 'Religions' title_en over the years using Matplotlib."],
    ["Generate Python code to create a line graph that specifically illustrates the trends for the 'General Information' title_en over the years."],
    ["What are the top 5 title_en that have most values in year 2014, and what are their values?"],
    ["What is the value when title_en is equal to 'Linguistics' in year '2016'?"],
    ["What are the 5 title_en that have least values in year 2016, and what are their values?"],
    ["Create Python code to generate a bar graph that visually represents different title_en values for the year 2015. Make sure that title_en names are clearly visible. Please utilize Matplotlib for this data visualization."],
]

llm_models = [
    "TheBloke/CodeLlama-7B-Python-GPTQ",
    "TheBloke/CodeLlama-13B-Python-GPTQ",
    "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
    "SaloniJhalani/ft-falcon-7b-instruct",
]

with gr.Blocks() as demo :

    gr.Markdown("# Omdena-UAE-LLM Demo")

    with gr.Row():

        with gr.Column():

            txt_url = gr.Textbox(label="Inference URL:", value=base_url)
            txt_query = gr.Textbox(label="Query:", value=examples[0][0])
            btn_submit = gr.Button(value="Submit")

        with gr.Column():
            txt_output = gr.Textbox(value="", label="Output")

    btn_submit.click(predict, inputs=[txt_url, txt_query], outputs=[txt_output])

    gr.Examples(
        examples=examples,
        inputs=txt_query,
        outputs=txt_output,
        fn=predict,
        cache_examples=True,
    )

demo.launch(server_name="0.0.0.0", server_port=8000)
