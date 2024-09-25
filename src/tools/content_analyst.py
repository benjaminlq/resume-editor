from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import LLM

from typing import Tuple, Union, Optional

CONTENT_CRITIQUE_LLM = OpenAI(model="o1-preview", temperature=0.2, max_completion_tokens=40000)

CV_CONTENT_CRITIQUE_PROMPT_WITH_JD = """You are an HR specialist with expertise in building effective resumes.
You are not afraid to constructively comment on the weak aspects of the resume. Be honest, do not make up information.
You will be given a job description and a resume. Based on the job description, critique the resume by focusing on the following:

- How well the resume highlights the required skills and qualifications.
- Areas where the resume could better align with the job description.
- Suggestions for enhancing the structure, formatting, or presentation.
- Any missing or underemphasized experiences or accomplishments that could strengthen the resume.
- Also analyse if there are unnecessary content which does not provide values to the resume with respect to the job description.

Be specific in your feedback and suggest actionable improvements. Also consider the job level of the resume and the job description. If you think that the resume is not suitable for the job, please explain why.

<START OF RESUME>
{resume}
</END OF RESUME>

<START OF JOB DESCRIPTION>
{job_description}
</END OF JOB DESCRIPTION>
"""

CV_CRITIQUE_PROMPT_TEMPLATE_WITH_JD = PromptTemplate(CV_CONTENT_CRITIQUE_PROMPT_WITH_JD)

CV_CONTENT_CRITIQUE_PROMPT_NO_JD = """You are an HR specialist with expertise in building effective resumes.
You are not afraid to constructively comment on the weak aspects of the resume. Be honest, do not make up information.
You will be given a resume. Critique the resume by focusing on the following:

- How well the resume highlights the required skills and qualifications.
- Suggestions for enhancing the structure, formatting, or presentation.
- Any missing or underemphasized experiences or accomplishments that could strengthen the resume.
- Also analyse if there are unnecessary content which does not provide values to the resume.

Be specific in your feedback and suggest actionable improvements. Also consider the job level of the resume.

<START OF RESUME>
{resume}
</END OF RESUME>
"""

CV_CRITIQUE_PROMPT_TEMPLATE_NO_JD = PromptTemplate(CV_CONTENT_CRITIQUE_PROMPT_NO_JD)

def critique_cv_content(
    resume: str,
    job_description: Optional[str] = None,
    llm: LLM = CONTENT_CRITIQUE_LLM,
    return_query: bool = False
) -> Union[Tuple[str, str], str]:
    if job_description:
        query = CV_CRITIQUE_PROMPT_TEMPLATE_WITH_JD.format(
            resume=resume, job_description=job_description
            )
    else:
        query = CV_CRITIQUE_PROMPT_TEMPLATE_NO_JD.format(resume=resume)
        
    response = llm.complete(query)
    
    return (query, response.text) if return_query else response.text