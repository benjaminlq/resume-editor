from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import LLM
from typing import Optional

CV_REVIEW_PROMPT_WITH_JD = """\You are a senior career advisor. You are given an original resume, (optionally) a job description and a critique on the strengths and weaknesses of the resume.
Your task is to use the critique to improve the resume. The improved version should address the weak points of the resume and implement the recommendations as needed.
The output should only contain the improved resume, nothing else. The improved resume should be formatted in formatted Markdown format.
{extra_instructions}

<START OF RESUME>
{resume}
<END OF RESUME>

<START OF JOB DESCRIPTION>
{job_description}
<END OF JOB DESCRIPTION>

<START OF CRITIQUE>
{critique}
<END OF CRITIQUE>

IMPROVED RESUME:
"""

CV_REVIEW_PROMPT_NO_JD = """\You are a responsible and honest senior career advisor. You are given an original resume, (optionally) a job description and a critique on the strengths and weaknesses of the resume.
Your task is to use the critique to improve the resume. The improved version should address the weak points of the resume and implement the recommendations as needed.
DO NOT make up facts that did not exist from the original resume.
The output should only contain the improved resume, nothing else. The improved resume should be formatted in formatted Markdown format.
{extra_instructions}

<START OF RESUME>
{resume}
<END OF RESUME>

<START OF CRITIQUE>
{critique}
<END OF CRITIQUE>

IMPROVED RESUME:
"""

CV_REVIEW_PROMPT_TEMPLATE_WITH_JD = PromptTemplate(CV_REVIEW_PROMPT_WITH_JD)
CV_REVIEW_PROMPT_TEMPLATE_NO_JD = PromptTemplate(CV_REVIEW_PROMPT_NO_JD)

def edit_cv(
    resume: str,
    critique: str,
    editor_llm: LLM,
    extra_instructions: str = "",
    job_description: Optional[str] = None,
) -> str:
    if job_description:
        query = CV_REVIEW_PROMPT_TEMPLATE_WITH_JD.format(
            resume=resume, critique=critique, job_description=job_description, extra_instructions=extra_instructions
            )
    else:
        query = CV_REVIEW_PROMPT_TEMPLATE_NO_JD.format(
            resume=resume, critique=critique, extra_instructions=extra_instructions
            )
        
    editted_cv = editor_llm.complete(query).text
    
    return editted_cv