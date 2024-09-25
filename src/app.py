import gradio as gr
import os

from utils import combine_documents, convert_llamaindex_messages_to_gradio
from pdf2image import convert_from_path
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import ChatMessage, MessageRole

from tools.jd_extractor import (
    refine_job_description,
    extract_job_description_from_url
)
from tools.content_analyst import critique_cv_content
from tools.layout_analyst import critique_cv_layout

from dotenv import load_dotenv
load_dotenv()

CHATBOT_LLM = OpenAI(model="gpt-4o", max_tokens=1024)
EXTRACTION_LLM = OpenAI(model="gpt-4o-mini", max_tokens=4096)
VISUAL_CRITIQUE_LLM = OpenAI(model="gpt-4o", max_tokens=4096)
CONTENT_CRITIQUE_LLM = OpenAI(model="o1-preview", max_completion_tokens=40000)

CHATBOT_SYSTEM_PROMPT = (
    "This is a conversation between a human and an AI. "
    "The AI is an honest and intelligent HR specialist with expertise in building effective resumes. "
    "If the AI does not know the answer, it will say 'I don't know' and will not make up information."
    )

with gr.Blocks(title="main") as demo:

    gr.Markdown("""# Title""")
    state = gr.State({
        "chat_messages": [
            ChatMessage(role=MessageRole.SYSTEM, content=CHATBOT_SYSTEM_PROMPT)
            ]
    })

    # File Upload
    ## Layout
    with gr.Tab(label="File Upload") as file_uploader_tab:
        with gr.Row():
            with gr.Column():
                cv_markdown = gr.Markdown("Please Upload your Resume to begin")
                cv_input = gr.File(file_count="single", type="filepath")
                cv_images = gr.Gallery(label="CV Preview")

                ## To delete
                # debug_text = gr.Textbox(label="Debug", lines=3)
                # debug_button = gr.Button("Debug")
                # @debug_button.click(inputs=cv_images, outputs=debug_text)
                # def debug(images):
                #     return {debug_text: str(images)}

            with gr.Column():
                jd_markdown = gr.Markdown("This is to fill in the job description")
                jd_layout_selector = gr.Radio(choices=["File Upload", "Text Description", "URL"], label="Choose input type", value="File Upload")                

                with gr.Column(visible=True) as file_upload:
                    jd_upload_button = gr.UploadButton()
                    jd_output_upload = gr.Textbox(label="Job Description", lines=10)
                    jd_clear_upload = gr.Button("Clear")

                with gr.Column(visible=False) as text_input:
                    jd_text_input = gr.Textbox(label="Job Description", lines=5)
                    jd_text_button = gr.Button("Submit")
                    jd_output_text = gr.Textbox(label="Job Description", lines=10)
                    jd_clear_text = gr.Button("Clear")
                
                with gr.Column(visible=False) as url_input:
                    jd_url_input = gr.Textbox(label="Job Description URL", lines=1)
                    jd_url_button = gr.Button("Submit")
                    jd_output_url = gr.Textbox(label="Job Description", lines=10)
                    jd_clear_url = gr.Button("Clear")

    ## Events
    @jd_upload_button.upload(inputs=[jd_upload_button, state], outputs=jd_output_upload)
    def upload_jd_file(jd_path, current_state):
        try:
            jd_data = combine_documents(
                SimpleDirectoryReader(input_files = [jd_path]).load_data()
                )
            current_state["jd_data"] = jd_data
            return {jd_output_upload: jd_data}
        except:
            current_state["jd_data"] = ""
            return {jd_output_upload: "Error: Please upload a valid .pdf or .docx file"}

    @jd_text_button.click(inputs=[jd_text_input, state], outputs=jd_output_text)
    def upload_jd_text(text, current_state):
        jd_data, success = refine_job_description(text)
        if success:
            current_state["jd_data"] = jd_data
        return {jd_output_text: jd_data}
    
    @jd_url_button.click(inputs=[jd_url_input, state], outputs=jd_output_url)
    def upload_jd_url(url, current_state):
        jd_data, success = extract_job_description_from_url(url)
        if success:
            current_state["jd_data"] = jd_data
        return {jd_output_url: jd_data}

    @gr.on(
        triggers = [jd_clear_upload.click, jd_clear_text.click, jd_clear_url.click],
        inputs = [jd_layout_selector, state],
        outputs = [jd_output_upload, jd_output_text, jd_output_url]
    )
    def clear_jd(selected_layout, current_state):
        if selected_layout == "File Upload":
            current_state["jd_data"] = ""
            return {jd_output_upload: ""}
        elif selected_layout == "Text Description":
            current_state["jd_data"] = ""
            return {jd_output_text: ""}
        elif selected_layout == "URL":
            current_state["jd_data"] = ""
            return {jd_output_url: ""}

    @jd_layout_selector.change(inputs=jd_layout_selector, outputs=[file_upload, text_input, url_input])
    def update_layout(selected_layout):
        if selected_layout == "File Upload":
            return {
                file_upload: gr.Column(visible=True),
                text_input: gr.Column(visible=False),
                url_input: gr.Column(visible=False),
            }

        elif selected_layout == "Text Description":
            return {
                file_upload: gr.Column(visible=False),
                text_input: gr.Column(visible=True),
                url_input: gr.Column(visible=False),
            }
            
        elif selected_layout == "URL":
            return {
                file_upload: gr.Column(visible=False),
                text_input: gr.Column(visible=False),
                url_input: gr.Column(visible=True),
            }
        
    @cv_input.upload(inputs=[cv_input, state], outputs=[cv_images, cv_markdown])
    def upload_cv(file_path, current_state):
        converted_images = convert_from_path(file_path, dpi=300)
        cv_data = combine_documents(
            SimpleDirectoryReader(input_files = [file_path]).load_data()
            )
        filename = os.path.basename(file_path)
        current_state["cv_data"] = cv_data
        current_state["cv_images"] = converted_images # List of PIL.Image
        return {cv_images: converted_images, cv_markdown: filename}

    @cv_input.clear(inputs=state, outputs=[cv_images, cv_markdown])
    def remove_cv(current_state):
        current_state["cv_data"] = ""
        current_state["cv_images"] = []
        return {cv_images: [], cv_markdown: "Please Upload your Resume to begin"}

    # Chatbot
    ## Layout
    with gr.Tab(label="Chatbot") as chatbot:
        gr.Markdown(CHATBOT_SYSTEM_PROMPT)
        chatbot = gr.Chatbot()
        chat_message = gr.Textbox(placeholder="Type your message here")
        
        with gr.Row():
            chat_button = gr.Button("Send")
            clear_message_button = gr.Button("Clear Message History")
        
        with gr.Row():
            analysis_button = gr.Button("Analyze resume", size="sm")
        
        with gr.Row():
            content_analysis = gr.Textbox(label="Content Analysis", lines=10)
            layout_analysis = gr.Textbox(label="Layout Analysis", lines=10)        
        
    def user_chat(user_message, current_state, evt_data: gr.EventData):
        messages = current_state["chat_messages"]
        gradio_messages = convert_llamaindex_messages_to_gradio(messages)
        messages.append(ChatMessage(role=MessageRole.USER, content=user_message))
        current_state["chat_messages"] = messages
        gradio_messages.append((user_message, None))
        
        return {chat_message: "", chatbot: gradio_messages}

    def ai_respond(current_state, evt_data: gr.EventData):
        messages = current_state["chat_messages"]
        response = CHATBOT_LLM.chat(messages)
        response_str = response.message.content
        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response_str))
        current_state["chat_messages"] = messages
        gradio_messages = convert_llamaindex_messages_to_gradio(messages)
        
        return {chat_message: "", chatbot: gradio_messages}

    gr.on(
        triggers = [chat_message.submit, chat_button.click],
        fn = user_chat,
        inputs = [chat_message, state],
        outputs = [chat_message, chatbot]
        ).then(
            fn = ai_respond,
            inputs = state,
            outputs = [chat_message, chatbot]
            )
    
    @clear_message_button.click(inputs=state, outputs=chatbot)
    def clear_chat_message(current_state):
        if current_state["chat_messages"][0].role == MessageRole.SYSTEM:
            current_state["chat_messages"] = [current_state["chat_messages"][0]]
        else:
            current_state["chat_messages"] = []
        return {chatbot: None}

    @analysis_button.click(inputs=state, outputs=[content_analysis, layout_analysis, chatbot])
    def analyze_resume(current_state):
        cv_data = current_state.get("cv_data", "")
        cv_images = current_state.get("cv_images", [])
        jd = current_state.get("", "")
        messages = current_state["chat_messages"]
        user_message = "Please help to analyze my resume."
        messages.append(ChatMessage(role=MessageRole.USER, content=user_message))
        
        if not cv_data or not cv_images:
            ai_message = "Resume not found. Please upload the resume first before I can perform analysis."
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=ai_message))
            current_state["chat_messages"] = messages
            gradio_messages = convert_llamaindex_messages_to_gradio(messages)
            return {content_analysis: "", layout_analysis: "", chatbot: gradio_messages}
        
        else:
            if jd:
                content_analysis_response = critique_cv_content(
                    resume=cv_data,
                    job_description=jd,
                    llm=CONTENT_CRITIQUE_LLM
                )
                layout_analysis_response = critique_cv_layout(
                    resume = cv_images,
                    job_description = jd,
                    llm=VISUAL_CRITIQUE_LLM
                )
                
            else:
                content_analysis_response = critique_cv_content(
                    resume=cv_data,
                    llm=CONTENT_CRITIQUE_LLM
                )
                layout_analysis_response = critique_cv_layout(
                    resume = cv_images,
                    llm=VISUAL_CRITIQUE_LLM
                )
                
            overall_analysis = f"# Content Analysis\n{content_analysis_response}\n\n\n # Layout Analysis\n{layout_analysis_response}\n"
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=overall_analysis))
            current_state["chat_messages"] = messages
            gradio_messages = convert_llamaindex_messages_to_gradio(messages)
            
            return {
                content_analysis: content_analysis_response,
                layout_analysis: layout_analysis_response,
                chatbot: gradio_messages
                }    

demo.launch()