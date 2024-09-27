import urllib

from retry import retry
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.indices import SummaryIndex
from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent

MAX_CHUNK_SIZE = 128000

JOB_EXTRACTION_QUERY = "Extract Job Information from the HTML text given under context. Return empty string if there is no job description found from the url"

JD_AGENT_SYSTEM_PROMPT = """You are an HR specialist with expertise in building effective resumes.
You will be given a job description in text, which may contain URL links to the job description and requirements.
If there are URLs relevant to describing job description and requirements, use the extraction tool to extract the information.
Append the relevant information collected from the URLs to the original job description only if the content extracted are relevant.

Your output only contains information about the job description and requirements, exclude any filler texts. If there is no relevant information, return 'Please provide a valid job description in as text or URL link'
"""

@retry(tries=5)
def extract_url(
    url: str
) -> str:
    f = urllib.request.urlopen(url)
    url_content = f.read()
    return url_content.decode('utf-8')

def extract_job_description_from_url(
    url: str
):
    """
    Use this function to extract the job description from the url
    """

    extraction_llm = OpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=4096)
    sentence_splitter = SentenceSplitter(chunk_size = MAX_CHUNK_SIZE)
    
    try:
        url_content = extract_url(url)
        jd_index = SummaryIndex.from_documents(
            documents=[Document(text=url_content)],
            transformations=[sentence_splitter.get_nodes_from_documents],
        )

        jd_extractor_query_engine = jd_index.as_query_engine(
            llm=extraction_llm
        )

        jd = jd_extractor_query_engine.query(JOB_EXTRACTION_QUERY).response
        success = True
        
    except:
        jd = "Job description cannot be extracted from given URL"
        success = False

    return jd, success

JD_EXTRACTION_TOOL = FunctionTool.from_defaults(
    fn=extract_job_description_from_url
)

def refine_job_description(
    job_description: str
):
    LLM = OpenAI(model="gpt-4o", max_tokens=4096)
    jd_extraction_agent = OpenAIAgent.from_tools(
        [JD_EXTRACTION_TOOL],
        llm=LLM,
        system_prompt=JD_AGENT_SYSTEM_PROMPT
    )
    
    jd_response = jd_extraction_agent.chat(job_description)
    if "please provide a valid job description" in jd_response.response.lower():
        success = False
    else:
        success = True

    return jd_response.response, success
    