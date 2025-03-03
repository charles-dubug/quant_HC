from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from pydantic import BaseModel
from prompts import news_cn_tech, news_cn_consumer, news_cn_entertainment, news_cn_policy, news_cn_social, news_us_stock

client = genai.Client(api_key="AIzaSyBg6j4CsCajX-eV_UTfaFLZnioBKS0n0EE")
model_id = "gemini-2.0-flash"

google_search_tool = Tool(
    google_search = GoogleSearch()
)

class News(BaseModel):
    event_type: str
    title: str
    source: str
    date: str
    event_abstract: str

response = client.models.generate_content(
    model=model_id,
    contents=news_us_stock,
    config=GenerateContentConfig(
        tools=[google_search_tool],
        response_modalities=["TEXT"],
    )
)

for each in response.candidates[0].content.parts:
    print(each.text)