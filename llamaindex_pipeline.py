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
import uuid
import json
import httpx
import openai
import weaviate
from pydantic import BaseModel
from six.moves.urllib.parse import urljoin
from typing import Optional, Any, Dict, Tuple
from llama_index.llms.openai_like import OpenAILike
from typing import List, Union, Generator, Iterator
from weaviate.config import AdditionalConfig, Timeout
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference


class CustomTextEmbeddingsInference(TextEmbeddingsInference):
    client: openai.OpenAI | None = None
    async_client: openai.AsyncOpenAI | None = None
    batch_size: int = 32

    def __init__(self, batch_size: int, api_key: str, default_headers: Dict[str, str], **kwargs):
        super().__init__(**kwargs)

        self.auth_token = api_key.strip()
        if self.auth_token.lower().startswith("bearer "):
            self.auth_token = self.auth_token[7:].strip()
        self.batch_size = batch_size
        self.timeout = 300
        #print(f"Initializing CustomTextEmbeddingsInference with base_url: {self.base_url}, auth_token: {self.auth_token}", flush=True)  
        try:
            self.client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.auth_token,
                http_client=httpx.Client(verify=False),
                default_headers=default_headers,
            )
            self.async_client = openai.AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.auth_token,
                http_client=httpx.AsyncClient(verify=False),
                default_headers=default_headers,
            )
        except Exception as e:
            # logging.error(f"Failed to initialize OpenAI client(s): {e}")
            raise str(e)

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(6):
            try:
                result = self.client.embeddings.create(
                    model=self.model_name,
                    extra_body={"input_type": "query"},
                    input=texts,
                )
                if hasattr(result, "response") and getattr(result.response, "status_code", 200) != 200:
                    raise Exception(f"Status code: {result.response.status_code}")
                return [item.embedding for item in result.data]
            except Exception as e:
                # logging.error(f"Embedding request failed: {e}, retrying {attempt+1}/6 after 5 seconds")
                time.sleep(5)
        raise Exception("Failed to get embeddings after 6 attempts")

    async def _acall_api(self, texts: List[str]) -> List[List[float]]:
        import asyncio
        for attempt in range(6):
            try:
                result = await self.async_client.embeddings.create(
                    model=self.model_name,
                    extra_body={"input_type": "query"},
                    input=texts,
                )
                if hasattr(result, "response") and getattr(result.response, "status_code", 200) != 200:
                    raise Exception(f"Status code: {result.response.status_code}")
                return [item.embedding for item in result.data]
            except Exception as e:
                # logging.error(f"Embedding request failed: {e}, retrying {attempt+1}/6 after 5 seconds")
                await asyncio.sleep(5)
        raise Exception("Failed to get embeddings after 6 attempts")


def get_openai_embedding(
    embedding_url: str, embedding_key: str,
) -> OpenAIEmbedding:
    return OpenAIEmbedding(model=embedding_url, api_key=embedding_key)


def get_model_id(client):
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


def fetch_model_info_from_url(emburl: str, embkey: str) -> Tuple[str, str]:
    model_name = ""
    model_type = ""
    
    headers = {}
    base_url = emburl
    
    # Remove "Bearer " prefix if present in emb_key
    if embkey.lower().startswith("bearer "):
        embkey = embkey.strip()[7:].strip()

    try:
        #print(f"fetch_model_info_from_url: base_url={base_url}, emb_key={emb_key}")
        with httpx.Client(verify=False) as httpx_client:
            openai_client = openai.OpenAI(
                base_url=base_url,
                api_key=embkey,
                default_headers=headers,
                http_client=httpx_client,
            )
            model_name = get_model_id(openai_client)
        model_type = "dkubex"
        # logging.debug(f"Embedding Model ID = {model_name}")
    except Exception as e:
        #logging.error(f"Error fetching model info: {e}")
        headers = {"Authorization": "Bearer " + embkey}
        # Remove "v1/" from the URL
        new_url = urljoin(emburl.replace("v1/", ""), "info")
        response = httpx.request(
            method="GET",
            url=new_url,
            follow_redirects=True,
            headers=headers,
            verify=False,
        )
        # logging.debug(f"Respone = {response}")
        model_name = response.json()["model_id"]
        # logging.debug(f"Embedding Model ID = {model_name}")
        model_type = "sky"
    
    return model_name, model_type


def get_emb_model_id(emburl: str, embkey: str, return_type: bool = False) -> str | Tuple[str, str]:
    model_name = ""
    model_type = ""
    
    assert emburl, f"Could not find embedding url. Invalid config for embedding in yaml"
    assert embkey, f"Could not find embedding key Invalid config for embedding in yaml"
    
    # logging.debug(f"Embedding Model-URL = {emburl}")
   
    # if not emburl.startswith("http"):
    #     try:
    #         model_name = get_openai_embedding(emburl, embkey).model_name
    #         model_type = "openai"
    #     except:
    #         model_name = emburl
    #         model_type = "huggingface"
    # else:
    
    try:
        model_name, model_type = fetch_model_info_from_url(emburl, embkey)
    except Exception as e:
        assert False, f"Provide correct end-point url of the deployed embedding model. {e}"

    if return_type:
        return model_name, model_type
    
    return model_name




class Pipeline:
    class Valves(BaseModel):
        llm_end_point: Optional[str] = "dummy"
        llm_api_key: Optional[str] = "dummy"
        emb_end_point: Optional[str] = "dummy"
        emb_api_key: Optional[str] = "dummy"
        enable_securellm: bool = False
        securellm_url: Optional[str] = "dummy"
        securellm_appkey: Optional[str] = "dummy"
        user_prompt: Optional[str] = "either-path-or-string"
        dataset: str = "dummy"
        textkey: str = "paperdocs"
        top_k: int = 3


    def __init__(self):
        self.embed_model: Any = None
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
                "user_prompt": os.getenv("USER_PROMPT", "either-path-or-string"),
            }
        )

        self.flag = 0

    def get_model_name(self, llmbase: str, llmkey: str, headers: Dict[str, Any]):
        with httpx.Client(verify=False) as _htx_cli:
            client = openai.OpenAI(
                base_url=llmbase,
                api_key=llmkey,
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
    
    def get_llm_data(self,) -> Tuple[str, str, Dict[str, Any]]:
        llmbase = ""
        llmkey = ""
        headers = {}

        if self.valves.enable_securellm:
            # llmbase = str(self.valves.securellm_url) + "/api/securellm/v1"
            llmbase = str(self.valves.securellm_url) + "/api/securellm"
            llmkey = "Bearer " + str(self.valves.securellm_appkey)
            
            headers.update(
                {
                    "x-request-id": str(uuid.uuid4()),
                    "x-sgpt-flow-id": str(uuid.uuid4()),
                    "X-Auth-Request-Email": os.getenv("USER", "anonymous"),
                    "llm-provider": self.valves.llm_end_point,
                    "Authorization": llmkey,
                }
            )

        else:
            headers = {"Authorization": self.valves.llm_api_key}
            llmbase = self.valves.llm_end_point
            llmkey = self.valves.llm_api_key

        return llmbase, llmkey, headers


    def get_chat_engine(self, ):
        llmbase, llmkey, headers = self.get_llm_data()

        _htx_cli = httpx.Client(verify=False) if llmbase.startswith("https") else None
        _htx_acli = (
            httpx.AsyncClient(verify=False) if llmbase.startswith("https") else None
        )

        tags = {
            "dataset": self.valves.dataset,
            "embedding_model": self.embed_model.model_name,
        }
        
        llm_model = self.get_model_name(llmbase, llmkey, headers)
        headers.update({"x-llm-tags": json.dumps(tags)})

        print(f"Headers : {headers}")
        print('**'*10)
        llm = OpenAILike(
            model=llm_model,
            api_base=llmbase,
            api_key=llmkey,
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
        text_data_list = [ 
                ( node.metadata.get("file_name", ""), node.metadata.get("file_path", ""), node.text )
                for node in self.nodes 
            ]

        context = ""
        for text_data in text_data_list:
            if text_data[0]:
                context += text_data[0]

            if text_data[1]:
                context += f" ({text_data[1]}): "

            if text_data[2]:
                context += f"{text_data[2]}\n"

        prompt = self.user_prompt.format(context=context, question=query)
        return prompt
       
    def get_weaviate_client(self):
        client = None
        try:
            http_host = os.getenv("WEAVIATE_SERVICE_HOST", "weaviate.d3x.svc.cluster.local")

            http_grpc_host = os.getenv(
                "WEAVIATE_SERVICE_HOST_GRPC", "weaviate-grpc.d3x.svc.cluster.local"
            )

            http_port = os.getenv("WEAVIATE_SERVICE_PORT", "80")
            grpc_port = os.getenv("WEAVIATE_GRPC_SERVICE_PORT", "50051")

            api_key = os.getenv("DKUBEX_API_KEY", "")

            if os.getenv("WEAVIATE_AUTH_APIKEY", None) != None:
                api_key = os.environ["WEAVIATE_AUTH_APIKEY"]

            additional_headers = {"authorization": f"Bearer {api_key}"}
            if "://" in http_host:
                http_host = http_host.split("://")[1]
            client = weaviate.connect_to_custom(
                http_host=http_host,
                http_port=int(http_port),
                http_secure=False if int(http_port) == 80 else True,
                grpc_host=http_grpc_host,
                grpc_port=int(grpc_port),
                grpc_secure=False if int(grpc_port) != 443 else True,
                headers=additional_headers,
                additional_config=AdditionalConfig(
                    timeout=Timeout(init=120, query=300, insert=300)  # Values in seconds
                ),
            )

            if not client.is_ready():
                raise ConnectionError("Weaviate instance not ready")

            return client

        except Exception as e:
            # logging.error(f"Error in creating Weaviate Client. Error: {str(e)}")
            if client:
                client.close()
            raise Exception(f"Error in creating Weaviate Client. Error: {str(e)}")

    async def on_startup(self):
        self.weaviate_client = self.get_weaviate_client()

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass
    
    def get_embed_model(
            # self, emburl: str, embkey: str, headers: Dict[str, Any]
            self, emburl: str, embkey: str
        ) -> CustomTextEmbeddingsInference:
        if not self.valves.emb_end_point.startswith("http"):
            try:
                return get_openai_embedding(self.valves.emb_end_point, self.valves.emb_api_key)
            except:
                try:
                    return HuggingFaceEmbedding(model_name=self.valves.emb_end_point)
                except Exception as e:
                    error_msg = f"Error while fetching embedding model. {e}"
                    raise Exception(error_msg)

        emb_model, model_type = get_emb_model_id(emburl, embkey, return_type=True)
        headers = {}
        try:
            emb_inference = CustomTextEmbeddingsInference(
                model_name=emb_model,
                base_url=emburl,
                text_instruction=" ",
                query_instruction=" ",
                truncate_text=False,
                batch_size=10,
                api_key=embkey,
                default_headers=headers
            )
        except Exception as e:
            error_msg = f"Error while fetching embedding model. {e}"
            raise Exception(error_msg)

        return emb_inference
    
    def get_emb_data(self,) -> Tuple[str, str, Dict[str, Any]]:
        emburl = ""
        embkey = ""
        headers = {}
        
        if self.valves.enable_securellm:
            emburl = str(plcfg["securellm"]["dkubex_url"]) + "/api/securellm"
            embkey = "Bearer " + str(plcfg["securellm"]["appkey"])
            
            headers.update(
                {
                    "llm-provider": self.valves.emb_end_point,
                    "Authorization": embkey,
                    "sllm-stream-response": "false",
                }
            )

        else:
            emburl = self.valves.emb_end_point
            embkey = self.valves.emb_api_key

        return emburl, embkey, headers

    def get_weaviate_retriever(self):
        if not self.embed_model:
            # emburl, embkey, headers = self.get_emb_data()
            self.embed_model = self.get_embed_model(self.valves.emb_end_point, self.valves.emb_api_key)

        if not self.retriever:
            vector_store = WeaviateVectorStore(
                            weaviate_client=self.weaviate_client, 
                            index_name=f"D{self.valves.dataset}chunks",
                            text_key=self.valves.textkey
                        )
            vector_index = VectorStoreIndex.from_vector_store(
                            vector_store, 
                            show_progress=True, 
                            embed_model=self.embed_model
                        )
            self.retriever = vector_index.as_retriever(similarity_top_k=self.valves.top_k)

    def get_nodes(self, user_message):
        if not user_message.startswith("### Task"):
            self.nodes = self.retriever.retrieve(user_message)

    def rewrite_query(self, user_message: str):
        if not user_message.startswith("### Task"):
            rewritten_prompt = self.get_rewritten_prompt()
            response = self.chat_engine.chat(prompt)

    def get_chat_history(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        # print(messages)
        # print(user_message)

        self.get_weaviate_retriever()
        self.get_chat_engine()
        self.get_prompt()

        self.get_nodes(user_message)
        prompt = self.set_prompt(user_message)

        response = self.chat_engine.stream_chat(prompt)

        return response.response_gen


