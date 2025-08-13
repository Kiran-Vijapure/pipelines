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
import httpx
import openai
import chromadb
from pydantic import BaseModel
# from schemas import OpenAIChatMessage
from typing import Optional, Any, Dict
from llama_index.llms.openai_like import OpenAILike
from typing import List, Union, Generator, Iterator
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.simple import SimpleChatEngine


class Pipeline:
    class Valves(BaseModel):
        llm_end_point: Optional[str] = "dummy-llm-end-point"
        llm_api_key: Optional[str] = "dummy-llm-api-key"
        # openai_apikey: Optional[str] = "dummy-openai-key"
        user_prompt: Optional[str] = "either-path-or-string"

    def __init__(self):
        self.index: Any = None
        self.retriever: Any = None
        self.user_prompt: str = None
        self.nodes: List[Any] = None
        self.documents: List[Any] = None
        self.chat_engine: SimpleChatEngine = None
        self.valves: Dict[str, Any] = self.Valves(
            **{
                "llm_end_point": os.getenv("LLM_END_POINT", "dummy-llm-end-point"),
                "llm_api_key": os.getenv("LLM_API_KEY", "dummy-llm-api-key"),
                # "openai_apikey": os.getenv("APIKEY", "dummy-openai-key"),
                "user_prompt": os.getenv("USER_PROMPT", "either-path-or-string")
            }
        )

    def get_model_name(self,):
        headers = {"Authorization": self.valves.llm_api_key}

        with httpx.Client(verify=False) as _htx_cli:
            client = openai.OpenAI(
                base_url=self.valves.llm_end_point,
                api_key=self.valves.llm_api_key,
                default_headers=headers,
                http_client=_htx_cli,
            )

            models = client.models.list()

            # Case 1: Check if model_id is directly accessible
            if hasattr(models, "model_id"):
                return models.model_id

            # Case 2: Check if it's in the dict format
            elif hasattr(models, "dict"):
                model_dict = models.dict()
                if "data" in model_dict and len(model_dict["data"]) > 0:
                    return model_dict["data"][0]["id"]

            # If neither case works, raise an exception
            raise ValueError(
                "Unable to retrieve model ID. API response format may have changed."
            )

    def get_chat_engine(self, ):
        llmbase = self.valves.llm_end_point
        headers = {"Authorization": self.valves.llm_api_key}

        _htx_cli = httpx.Client(verify=False) if llmbase.startswith("https") else None
        _htx_acli = (
            httpx.AsyncClient(verify=False) if llmbase.startswith("https") else None
        )

        llm = OpenAILike(
            model=self.get_model_name(),
            api_base=llmbase,
            api_key=self.valves.llm_api_key,
            temperature=0,
            is_chat_model=True,
            # max_tokens=max_tokens,
            max_tokens=2048,
            default_headers=headers,
            http_client=_htx_cli,
            async_http_client=_htx_acli,
        )

        self.chat_engine = SimpleChatEngine.from_defaults(llm=llm)


    def get_prompt(self):
        try:
            with open(self.valves.user_prompt, "r") as f:
                self.user_prompt = f.read()
        except:
            self.user_prompt = self.valves.user_prompt

    def set_prompt(self,query: str) -> str:
        text_list = [ node.text for node in self.nodes ]
        context = "\n".join(text_list)
        prompt = self.user_prompt.format(context=context, question=query)
        return prompt
        

    async def on_startup(self):

        # Set the OpenAI API key
        # os.environ["OPENAI_API_KEY"] = os.getenv("APIKEY")
        # os.environ["OPENAI_API_KEY"] = self.valves.openai_apikey
        # weaviate_uri: Optional[str] = os.getenv("WEAVIATE_URI", "dummy-uri")
        # weaviate_auth_apikey: Optional[str] = os.getenv("WEAVIATE_AUTH_APIKEY", "dummy-auth-key"),

        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_APIKEY")
        self.documents = SimpleDirectoryReader("/home/kiran-vijapure/pipelines/maha").load_data()

        self.index = VectorStoreIndex.from_documents(self.documents)
        self.retriever = self.index.as_retriever(similarity_top_k=3)


    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def get_nodes(self, user_message):
        if not user_message.startswith("### Task"):
            self.nodes = self.retriever.retrieve(user_message)

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        # print(messages)
        # print(user_message)

        self.get_chat_engine()
        self.get_prompt()

        self.get_nodes(user_message)
        prompt = self.set_prompt(user_message)

        # query_engine = self.index.as_query_engine(streaming=True)
        # response = query_engine.query(user_message)

        response = self.chat_engine.stream_chat(prompt)

        return response.response_gen


