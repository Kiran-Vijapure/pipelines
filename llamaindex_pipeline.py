"""
title: Llama Index Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index
"""

import os
from pydantic import BaseModel
from schemas import OpenAIChatMessage
from typing import List, Union, Generator, Iterator
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None

    class Valves(BaseModel):
        llm_end_point: str = None
        llm_api_key: str = None

    async def on_startup(self):

        # Set the OpenAI API key
        os.environ["OPENAI_API_KEY"] = os.getenv("APIKEY")


        self.documents = SimpleDirectoryReader("/app/maha").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        # This function is called when the server is started.
        pass

        # Intialize llm
        # Get weaviate vectorstore client


    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        self.valves = self.Valves(
            **{
                "llm_end_point": os.getenv("LLM_END_POINT", "llm-end-point"),
                "llm_api_key": os.getenv("LLM_API_KEY", "llm-api-key")
            }
        )

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen


        # Get documents
        # Create prompt
        # Query to llm 
        # return response.
