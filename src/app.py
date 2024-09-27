import gradio as gr
import os
import asyncio
import openai

from utils import combine_documents, convert_llamaindex_messages_to_gradio
from pdf2image import convert_from_path
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import ChatMessage, MessageRole

from tools.jd_extractor import (
    refine_job_description,
    extract_job_description_from_url
)
from tools.content_analyst import acritique_cv_content
from tools.layout_analyst import acritique_cv_layout
from tools.editor import edit_cv

def _resolve_openai_model():
    client = openai.OpenAI(
        api_key = os.getenv('OPENAI_API_KEY')
    )   
    models = client.models.list()
    model_list = [model.id for model in models]
    if "o1-preview" not in model_list:
        return OpenAI(model="gpt-4o-mini", max_tokens=4096), OpenAI(model="gpt-4o-mini", max_tokens=4096)
    else:
        return OpenAI(model="o1-preview", max_completion_tokens=40000), OpenAI(model="o1-preview", max_completion_tokens=50000)

CHATBOT_SYSTEM_PROMPT = (
    "This is a conversation between a human and an AI. "
    "The AI is an honest and intelligent HR specialist with expertise in building effective resumes. "
    "If the AI does not know the answer, it will say 'I don't know' and will not make up information."
    )

with gr.Blocks(title="main") as demo:

    with gr.Column(visible=True) as login_block:
        gr.Markdown("### Enter your API Key to proceed")
        api_key_input = gr.Textbox(label="API Key", placeholder="Enter your API Key here")
        submit_button = gr.Button("Submit")
        error_message = gr.Textbox(visible=False)

    with gr.Column(visible=False) as main_block:
        
        try:
            CHATBOT_LLM = OpenAI(model="gpt-4o", max_tokens=1024)
            EXTRACTION_LLM = OpenAI(model="gpt-4o-mini", max_tokens=4096)
            VISUAL_CRITIQUE_LLM = OpenAI(model="gpt-4o", max_tokens=4096)
            CONTENT_CRITIQUE_LLM, EDITOR_LLM = _resolve_openai_model()
        except:
            pass    
                
        gr.Markdown("""# Resume Critique Bot\n### Powered by OpenAI o1-preview model""")
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
        ### Upload JD Events
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
        
        ### Upload CV Events
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
        with gr.Tab(label="Chatbot") as chatbot_tab:
            gr.Markdown("## Chatbot")
            chatbot = gr.Chatbot()
            chat_message = gr.Textbox(placeholder="Type your message here")
            
            with gr.Row():
                chat_button = gr.Button("Send")
                clear_message_button = gr.Button("Clear Message History")
            
            with gr.Row():
                analysis_button = gr.Button("Analyze resume", size="sm")
            
            with gr.Row():
                content_analysis = gr.Markdown(label="Content Analysis")
                layout_analysis = gr.Markdown(label="Layout Analysis")      

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
            jd = current_state.get("jd_data", "")
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
                    tasks = [
                        acritique_cv_content(
                            resume=cv_data,
                            job_description=jd,
                            llm=VISUAL_CRITIQUE_LLM # CONTENT_CRITIQUE_LLM
                            ),
                        acritique_cv_layout(
                            resume = cv_images,
                            job_description = jd,
                            llm=VISUAL_CRITIQUE_LLM
                            )
                        ]
                    
                else:
                    tasks = [
                        acritique_cv_content(
                            resume=cv_data,
                            llm=VISUAL_CRITIQUE_LLM #CONTENT_CRITIQUE_LLM
                            ),
                        acritique_cv_layout(
                            resume = cv_images,
                            llm=VISUAL_CRITIQUE_LLM
                            )
                        ]
                
                async def collect_critique(tasks):
                    return await asyncio.gather(*tasks)
                
                content_analysis_response, layout_analysis_response = asyncio.run(collect_critique(tasks))
                
                overall_analysis = f"# Content Analysis\n{content_analysis_response}\n\n\n # Layout Analysis\n{layout_analysis_response}\n"
                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=overall_analysis))
                current_state["chat_messages"] = messages
                current_state["overall_analysis"] = overall_analysis
                gradio_messages = convert_llamaindex_messages_to_gradio(messages)
                
                return {
                    content_analysis: "# Content Analysis:\n" + content_analysis_response,
                    layout_analysis: "# Layout Analysis:\n" +  layout_analysis_response,
                    chatbot: gradio_messages
                    }    

        # CV Editor 
        ## Layout
        with gr.Tab(label="Editor") as editor_tab:
            extra_inst = gr.Textbox(value="", label="Extra Instructions", lines=3)
            editor_button = gr.Button("Revise")
            editted_resume = gr.Markdown(label="Your Editted CV")
            ## Events
            @editor_button.click(inputs=[extra_inst, state], outputs=editted_resume)
            def edit_resume(extra_instructions, current_state):
                cv_data = current_state.get("cv_data", "")
                critique = current_state.get("overall_analysis", "")
                job_description = current_state.get("jd_data", "")
                if not cv_data:
                    return "Resume or not found. Please upload the resume first before I revise the resume."
                if not critique:
                    return "Please analyze the resume first before I can revise the resume."

                editted_cv = edit_cv(
                    resume=cv_data,
                    critique=critique,
                    extra_instructions=extra_instructions,
                    job_description=job_description,
                    editor_llm=EDITOR_LLM
                    )

                return editted_cv

    @submit_button.click(inputs=api_key_input, outputs=[login_block, main_block, error_message])
    def validate_api_key(api_key):
        try:
            client = openai.OpenAI(api_key = api_key)
            _ = client.models.list()
            os.environ["OPENAI_API_KEY"] = api_key
            return {login_block: gr.Column(visible=False), main_block: gr.Column(visible=True), error_message: gr.Textbox(visible=False)}
        except:
            return {login_block: gr.Column(visible=True), main_block: gr.Column(visible=False), error_message: gr.Textbox(visible=True, value="Invalid API Key. Please try again.")}

demo.launch()