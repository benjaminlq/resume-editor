from llama_index.multi_modal_llms.openai import OpenAIMultiModal
import io
import base64
from typing import Optional
from llama_index.core.schema import ImageDocument
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage, MessageRole
from llama_index.multi_modal_llms.openai.utils import generate_openai_multi_modal_chat_message

LAYOUT_CRITIQUE_LLM = OpenAIMultiModal(model="gpt-4o", temperature=0.2, max_new_tokens=4096, image_detail="high")

CV_LAYOUT_CRITIQUE_SYSTEM_PROMPT = """You are an honest and reliable HR specialist with expertise in building effective resumes.
You are not afraid to constructively comment on the weak aspects of the resume. Be honest, do not make up information.
You will be given a resume and optionally a job description. Your task is to critique the aesthetic aspects of the resume by focusing on the following:

1. Layout and Structure:
- Clarity and Organization: Use a clear, logical structure with well-defined sections.
- White Space: Incorporate sufficient white space between sections to avoid clutter and make the document breathable.
- Alignment: Maintain consistent alignment for headings, bullet points, and text blocks.
- Margins: Keep balanced margins for a clean and organized look.

2. Font choice, size and consistency:  Use professional, easy-to-read fonts, appropriate font size for headings and text contents. Use the same font throughout, reserving bold and italics for emphasis. Use font size, bolding, and spacing to guide the reader’s eye through the resume, ensuring the most important information (e.g., your name, job titles) stands out.

3. Color Scheme: Choice of color for texts, highlights, headings, etc.

4. Visual Elements: The usage appropriateness of icons or graphics. Design suitability for the job description and professionalism.

5. Use of Bold and Italics to highlight important information such as names, job titles, section headers

6. Consistency: Consistent formatting for dates, locations, and bullet points across different sections. Keep equal spacing between headings and paragraphs to ensure readability.

7. Length and Page Breaks: Whether the resume's length is appropriate. Are page breaks clean between sections, without splitting information awkwardly across pages.

8. Scannability:
Bullet Points: Use bullet points to break up large blocks of text, making it easy for the recruiter to scan.
Short Sentences: Keep sentences concise to improve readability.

Be specific in your feedback. If possible, suggest actionable improvements, only if the improvements have not been done by the original resume.
"""

def convert_PIL_to_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return base64_image

def critique_cv_layout(
    resume,
    job_description: Optional[str] = None,
    llm: LLM = LAYOUT_CRITIQUE_LLM,
):
    if not isinstance(resume, list):
        resume = [resume]
    
    image_documents = [
        ImageDocument(image=convert_PIL_to_base64(cv_image)) for cv_image in resume
    ]

    messages = [
        ChatMessage(content=CV_LAYOUT_CRITIQUE_SYSTEM_PROMPT, role=MessageRole.SYSTEM)
    ]
    
    if job_description:
        messages.append(
            ChatMessage(content=f"# Job description:\n\n{job_description}", role=MessageRole.SYSTEM)
        )
    
    messages.append(
        generate_openai_multi_modal_chat_message(
            prompt = "resume",
            role = "user",
            image_documents=image_documents,
            image_detail="high"
            )
    )
    
    response = llm.chat(messages)
    return response.message.content

async def acritique_cv_layout(
    resume: str,
    job_description: Optional[str] = None,
    llm: LLM = LAYOUT_CRITIQUE_LLM
):  
    if not isinstance(resume, list):
        resume = [resume]
    
    image_documents = [
        ImageDocument(image=convert_PIL_to_base64(cv_image)) for cv_image in resume
    ]

    messages = [
        ChatMessage(content=CV_LAYOUT_CRITIQUE_SYSTEM_PROMPT, role=MessageRole.SYSTEM)
    ]
    
    if job_description:
        messages.append(
            ChatMessage(content=f"# Job description:\n\n{job_description}", role=MessageRole.SYSTEM)
        )
    
    messages.append(
        generate_openai_multi_modal_chat_message(
            prompt = "resume",
            role = "user",
            image_documents=image_documents,
            image_detail="high"
            )
    )
    
    response = await llm.achat(messages)
    return response.message.content