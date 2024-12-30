#!/usr/bin/env python

import os, json
from typing import List
from operator import itemgetter

from fastapi import FastAPI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser

from langserve import add_routes

from dotenv import load_dotenv

import boto3
from botocore.exceptions import ClientError

load_dotenv()

def get_secret():

    secret_name = "prod/openai"
    region_name = "us-east-1"

    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e

    secret = get_secret_value_response['SecretString']
    keys = json.loads(secret)
    os.environ["LANGCHAIN_API_KEY"] = keys.get("LANGCHAIN_API_KEY")
    os.environ["OPENAI_API_KEY"] = keys.get("OPENAI_API_KEY")

get_secret()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using LangChain's Runnable interfaces",
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

class Input(BaseModel):
    document_text: str
    keywords: List[str]


class Output(BaseModel):
    document_text: str
    keywords: List[str]

parser = JsonOutputParser(pydantic_object=Output)

prompt = PromptTemplate(
    input_variables=["document_text", "keywords"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    template="""
        You are a helpful keyword search assistant.  Your goal is to find keywords in the provided document.  
        When the keyword is found, highlight the keyword with the <mark> tag in the document.  Also, keep a list of the found keywords.  Return the updated document and keywords in the JSON format.
        Document: {document_text}
        Keywords: {keywords}
        Output: {format_instructions}
    """
)

def remove_escape(text):
    return text.replace('\"', '"')

chain = (
    {
        "document_text": itemgetter("document_text") | RunnableLambda(remove_escape),
        "keywords": itemgetter("keywords")
    } 
    | prompt 
    | llm 
    | parser
)

add_routes(
    app, chain.with_types(input_type=Input), path="/keyword-processor"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)