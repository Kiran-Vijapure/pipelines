import os
import uuid
import time
import queue
import asyncio
import requests
from mlflow import MlflowClient
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from threading import Thread


mlflow_data_queue = queue.Queue()
client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI"))
shutdown_begins = False


def dict_log(client: MlflowClient, run_id: str, data: Dict[str, Any], file_name: str):
    client.log_dict(
        run_id=run_id,
        dictionary=data,
        artifact_file=file_name,
    )


def add_exp_permission(experiment_id):
    payload = {
        "permission": "EDIT",
        "id": experiment_id,
        "all_users": True,
        "publish": True,
    }

    # dkubex_url = os.getenv("DKUBEX_ADDERESS", "")
    dkubex_url = os.getenv(
        "DKUBEX_URL",
        os.getenv(
            "DKUBEX_ADDERESS",
            "http://ingress-nginx-controller.d3x.svc.cluster.local:80",
        ),
    )
    api_prefix = "d3x"
    headers = {
        "authorization": f"Bearer {os.getenv('APIKEY', os.getenv('DKUBEX_APIKEY', ''))}"
    }
    resp = requests.post(
        # f"{obj.url}/api/{obj.api_prefix}/mlflow/experiments/permission",
        f"{dkubex_url}/api/{api_prefix}/mlflow/experiments/permission",
        # headers=obj.headers,
        headers=headers,
        json=payload,
        verify=False,
    )
    resp.raise_for_status()


def get_run_id(client: MlflowClient, exp_id: str, run_name: str, tags):
    # Search dataset run if not then create run and return run id.
    run_id = ""

    run_list = client.search_runs(
        experiment_ids=[exp_id], filter_string=f"run_name='{run_name}'"
    ).to_list()

    if not run_list:
        # Create dataset run and get run id
        run_id = client.create_run(
            experiment_id=exp_id,
            run_name=run_name,
            tags=tags,
        ).info.run_id
        # run_id = run.info.run_id
    else:
        # Get dataset run id
        run_id = run_list[0].info.run_id

    return run_id


def get_question_run_id(
    client: MlflowClient, experiment_name: str, dataset_run_name: str
):
    exp = client.get_experiment_by_name(name=experiment_name)

    if not exp:
        exp_id = client.create_experiment(name=experiment_name)
        add_exp_permission(exp_id)
    else:
        exp_id = exp.experiment_id

    # Get dataset run id.
    dataset_run_id = get_run_id(client, exp_id, dataset_run_name, tags={})

    # Get conversation run id.
    conv_run_id = get_run_id(
        client, exp_id, "conv-default", tags={"mlflow.parentRunId": dataset_run_id}
    )

    # Get question run id
    question_run_id = get_run_id(
        client,
        exp_id,
        f"question-{str(uuid.uuid4())}",
        tags={"mlflow.parentRunId": conv_run_id},
    )

    return question_run_id


async def log_to_mlflow(data: Dict[str, Any]):
    global client

    ques_run_id = get_question_run_id(
        client, data["experiment_name"], data["dataset"] + "-ragquery"
    )

    # Log nodes
    dict_log(
        client,
        ques_run_id,
        {"reference_nodes": data["nodes"]},
        "reference_nodes.json",
    )

    # Need to add mlflow dataset.

    # Log questions
    dict_log(
        client,
        ques_run_id,
        {
            "question": data["question"],
        },
        "question.json",
    )

    # Log response
    dict_log(client, ques_run_id, {"response": data["answer"]}, "answer.json")

    # Log references
    dict_log(
        client,
        ques_run_id,
        {
            "response_references": data["reference_list"],
        },
        "answer_references.json",
    )

    # Log valves / config
    dict_log(client, ques_run_id, data["config"], "rag_config.json")


def process_mlflow_data():
    global shutdown_begins
    global mlflow_data_queue

    while True:
        if not mlflow_data_queue.empty():
            asyncio.run(log_to_mlflow(mlflow_data_queue.get()))

        if shutdown_begins:
            break


@asynccontextmanager
async def lifespan(app: FastAPI):
    global shutdown_begins
    task_thread = Thread(target=process_mlflow_data, daemon=True)
    task_thread.start()
    yield
    shutdown_begins = True
    task_thread.join()


app = FastAPI(lifespan=lifespan)


@app.post("/mlflow_log")
async def log_mlflow(request: Request):
    global mlflow_data_queue

    data = await request.json()
    mlflow_data_queue.put(data)

    return {"message": "MLflow data added to the queue"}
