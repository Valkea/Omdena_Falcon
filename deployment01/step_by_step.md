# Title

## Step 1: Set up Falcon 7B using Truss

### 1. Create a virtual environment

```code
>>> python3 -m venv venv_falcon7B
>>> source venv_falcon7B/bin/activate
```

### 2. Install Truss

```code
>>> pip3 install truss
```

### 3. Setup the working dir with Truss

```code
>>> truss init falcon_7b_truss
```

( I used `Falcon-7B` as model name when asked by the Truss setup tool )


### 4. Write the `model.py`

```python
"""
This script contains the implementation of a language model 
using the Falcon-7B-Instruct model from Hugging Face.
"""

#python code for the model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict

MODEL_NAME = "tiiuae/falcon-7b-instruct"
DEFAULT_MAX_LENGTH = 128


class Model:
    def __init__(self, data_dir: str, config: Dict, **kwargs) -> None:
        self._data_dir = data_dir
        self._config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("DEVICE INFERENCE RUNNING ON : ", self.device)
        self.tokenizer = None
        self.pipeline = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model_8bit = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True)

        self.pipeline = pipeline(
            "text-generation",
            model=model_8bit,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def predict(self, request: Dict) -> Dict:
        with torch.no_grad():
            try:
                prompt = request.pop("prompt")
                data = self.pipeline(
                    prompt,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_length=DEFAULT_MAX_LENGTH,
                    **request
                )[0]
                return {"data": data}

            except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}

```


## Step 2: Containerize the model and run it using docker

### 1. Write the `config.yaml`

```yaml
apply_library_patches: true
base_image: null
bundled_packages_dir: packages
data_dir: data
description: null
environment_variables: {}
examples_filename: examples.yaml
external_data: null
external_package_dirs: []
hf_cache: null
input_type: Any
live_reload: false
model_class_filename: model.py
model_class_name: Model
model_framework: custom
model_metadata: {}
model_module_dir: model
model_name: Falcon-7B
model_type: custom
python_version: py39
requirements:
- torch
- peft
- sentencepiece
- accelerate
- bitsandbytes
- einops
- scipy
- git+https://github.com/huggingface/transformers.git
resources:
  accelerator: null
  cpu: '3'
  memory: 14Gi
  use_gpu: true
runtime:
  predict_concurrency: 1
secrets: {}
spec_version: '2.0'
system_packages: []
train:
  resources:
    accelerator: null
    cpu: 500m
    memory: 512Mi
    use_gpu: false
  training_class_filename: train.py
  training_class_name: Train
  training_module_dir: train
  variables: {}

```

### 2. Write `main.py` to instruct Truss on how to bundle everything together

Place the `main.py` file in the same directory as the falcon_7b_truss folder.

```python
import truss
from pathlib import Path
import requests

tr = truss.load("./falcon_7b_truss")
command = tr.docker_build_setup(build_dir=Path("./falcon_7b_truss"))
print(command)
```

### 3. Build the Docker image

```cmd
>>> docker build falcon_7b_truss -t falcon-7b-model:latest
```

### 4. Push to a Cloud registry (GCP, AWS, Azure, Docker-Hub...)

#### 4.1 Let's use the Google Cloud Artifact Registry.

- Go to the GCP console and find the `Artifact Registry API` (enable it if it's not enabled yet)
- Click `CREATE REPOSITORY`
- Fill the fields:
    - name: omdena-falcon-repo
    - format: Docker
    - region: me-central1 (Doha)
- Click `CREATE`

#### 4.2 Install `gcloud` if not installed
https://cloud.google.com/sdk/docs/install


#### 4.3 Push the image to the repo

```cmd
>>> gcloud auth configure-docker <gcp-region>-docker.pkg.dev
>>> docker tag <locally build model name> <gcp-region>-docker.pkg.dev/<gcp-project-id>/<artifact-repo-name>/falcon-7b-model:latest
>>> docker push <gcp-region>-docker.pkg.dev/<project-id>/<artifact-repo-name>/falcon-7b-model:latest
``` 

You can get the `<gcp-region>-docker.pkg.dev/<gcp-project-id>/<artifact-repo-name>` on the repository details page of the repo we just created. Here is an example with all the values replaced with names previously used and my own gcp-project-id.

```cmd
>>> gcloud auth configure-docker me-central1-docker.pkg.dev
>>> docker tag falcon-7b-model me-central1-docker.pkg.dev/omdena-401512/omdena-falcon-repo/falcon-7b-model:latest
>>> docker push me-central1-docker.pkg.dev/omdena-401512/omdena-falcon-repo/falcon-7b-model:latest
``` 

## Step 3: Deploy the backend (model / API)

### 1. Generate Kubernetes cluster

```
>>> gcloud config set project <gcp_prject_id>
```

Let's replace the variables with ours
```
>>> gcloud config set project omdena-401512
```

```
>>> gcloud compute networks create omdena-playground-vpc \
    --subnet-mode=auto \
    --bgp-routing-mode=regional \
    --mtu=1460
```


```cmd
>>> gcloud beta container \
--project "<gcp-project-id>" clusters create "omdena-gpu-cluster-1" \
--zone "me-west1-c" \
--no-enable-basic-auth \
--cluster-version "1.27.3-gke.1 ResponseError: code=400, message=Network "omdena-playground-vpc" does not exist700" \
--release-channel "regular" \
--machine-type "n1-standard-4" \
--accelerator "type=nvidia-tesla-t4,count=1" \
--image-type "COS_CONTAINERD" \
--disk-type "pd-balanced" \
--disk-size "50" \
--metadata disable-legacy-endpoints=true \
--scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
--num-nodes "1" \
--logging=SYSTEM,WORKLOAD \
--monitoring=SYSTEM \
--enable-ip-alias \
--network "projects/<gcp-project-id>/global/networks/omdena-playground-vpc" \
--subnetwork "projects/<gcp-project-id>/regions/me-west1/subnetworks/omdena-playground-vpc" \
--no-enable-intra-node-visibility \
--default-max-pods-per-node "110" \
--security-posture=standard \
--workload-vulnerability-scanning=disabled \
--no-enable-master-authorized-networks \
--addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
--enable-autoupgrade \
--enable-autorepair \
--max-surge-upgrade 1 \
--max-unavailable-upgrade 0 \
--enable-managed-prometheus \
--enable-shielded-nodes \
--node-locations "me-west1-c"
```

Let's replace the variables with ours
```cmd
>>> gcloud beta container \
--project "compute.googleapis.com/gpus_all_regions" clusters create "omdena-gpu-cluster-1" \
--zone "me-west1-c" \
--no-enable-basic-auth \
--cluster-version "1.27.3-gke.1700" \
--release-channel "regular" \
--machine-type "n1-standard-4" \
--accelerator "type=nvidia-tesla-t4,count=1" \
--image-type "COS_CONTAINERD" \
--disk-type "pd-balanced" \
--disk-size "50" \
--metadata disable-legacy-endpoints=true \
--scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
--num-nodes "1" \
--logging=SYSTEM,WORKLOAD \
--monitoring=SYSTEM \
--enable-ip-alias \
--network "projects/omdena-401512/global/networks/omdena-playground-vpc" \
--subnetwork "projects/omdena-401512/regions/me-west1/subnetworks/omdena-playground-vpc" \
--no-enable-intra-node-visibility \
--default-max-pods-per-node "110" \
--security-posture=standard \
--workload-vulnerability-scanning=disabled \
--no-enable-master-authorized-networks \
--addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
--enable-autoupgrade \
--enable-autorepair \
--max-surge-upgrade 1 \
--max-unavailable-upgrade 0 \
--enable-managed-prometheus \
--enable-shielded-nodes \
--node-locations "me-west1-c"
```


### 2. Run Kubernetes node and service

Place the `truss-falcon-deployment.yaml` file in the same directory as the falcon_7b_truss folder.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: truss-falcon-7b
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      component: truss-falcon-7b-layer
  template:
    metadata:
      labels:
        component: truss-falcon-7b-layer
    spec:
      containers:
      - name: truss-falcon-7b-container
        image: <gcp-region>-docker.pkg.dev/<project-id>/<artifact-repo-name>/falcon-7b-model:latest
        ports:
          - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
```

Place the `truss-falcon-ilb-service.yaml` file in the same directory as the falcon_7b_truss folder.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: truss-falcon-7b-service-ilb
  annotations:
    networking.gke.io/load-balancer-type: "Internal"
    networking.gke.io/internal-load-balancer-allow-global-access: "true"
spec:
  type: LoadBalancer
  externalTrafficPolicy: Cluster
  selector:
    component: truss-falcon-7b-layer
  ports:
  - name: tcp-port
    protocol: TCP
    port: 80
    targetPort: 8080
```

Let's create the deployment and the service

```cmd
>>> kubectl create -f omdena-truss-falcon-deployment.yaml
>>> kubectl create -f omdena-truss-falcon-ilb-service.yaml
```

### 3. Getting information about Kubernetes deployments

After you run a Kubernetes cluster, you can query it to see information.

> The Deployment details
> ```cmd
> >>> kubectl get deployments
> ```
> 
> The Pod details
> ```cmd
> >>> kubectl get pods
> ```
> 
> The Service details
> ```cmd
> >>> kubectl get svc
> ```
> 
> The Virtual Service details
> ```cmd
> >>> kubectl get vs
> ```
> 
> The log details
> ```cmd
> >>> kubectl logs <pod id> <container id>
> ```


In order to get the deployment status, one need to fist get the pod name, then use this name to gain access to the logs.

1. Let's first get the pods list:
```cmd
>>> kubectl get pods
```

2. Then we can use the returned pod-id to access the logs and check if everything is going as expected with the following command:
```cmd
>>> kubectl logs <pod-id>`
```

And once the model is up and running, one needs to get the **name** and **IP** of the `LoadBalancer` service in order to access it. We can get them using the following command:
```cmd
>>> kubectl get svc
```

> Warning: One can access the API enpoint using the mentioned IP **only from a VM hosted in the same VPC**.
>
> And by default the port is :80 in Truss, but it might be different if you used another way to expose the model.

## Step 4: Deploy the Frontend

### 1. Define Serverless VPC Access

1. In GCP, go to `VPC network` -> `Serverless VPC access`, enable it if needed and click `CREATE CONNECTOR`.
2. Choose a name (i.e. `falcon7b-service-connector`)
3. Choose a region (you will need to use the **same region** for the `Cloud Run` later).
4. Choose the project's network (we used `omdena-playground-vpc` earlier).
5. Choose a `Custom IP range subnet` and input `10.8.0.0/28` as IP range.
6. Change instance type and number if needed, but the default values should be fine.

At this point, we now have a mapping from `falcon7b-service-connector.<project_name>.internal` to the LoadBalancer, but you can directly use the LoadBalancer IP.


### 2. Write a Gradio app (you can also use Streamlit of any other frontend)

Let's create an app.py file with the following content

```python
import gradio as gr
import requests

base_url = "THE LoadBalancer IP collected earlier. Something like 10.x.x.x:80 or <service_name>.<project_name>.internal"

def predict(question):
    data = {"prompt": question}	
    print("Infering...")
    res=requests.post(f"{base_url}/v1/models/model:predict", json=data)
    print(res.json())
    return res.json()

examples = [
    ["What does Omdena"],
    ["What is the Falcon LLM model"],
]

demo = gr.Interface(
    predict, 
    [ gr.Textbox(label="Enter prompt:", value="Can LLMs make coffee?"),
      
    ],
    "text",
    examples=examples,
    title= "Falcon7B demo"
    )

demo.launch(server_name="0.0.0.0", server_port=8000)
```

Be sure to use the `LoadBalancer` service's IP collected earlier in the `base_url` variable (ot the <service_name>.<project_name>.internal)

### 3. Let's containerize the frontend app

Here is the requirements.txt file 

```
gradio==3.37.0
Flask==2.2.2
requests==2.31.0
```

And here is Dockerfile

```dockerfile
# python3.8 breaks with gradio
FROM python:3.9

# Install dependencies from requirements.txt
COPY ./src/requirements.txt .
RUN pip install -r requirements.txt

COPY ./src /src

WORKDIR /src

EXPOSE 8000

CMD ["python", "app.py"]
```

Now let's build the docker image
```cmd
>>> gcloud auth configure-docker <gcp-region>-docker.pkg.dev
>>> gcloud builds submit --tag=<gcp-region>-docker.pkg.dev/<gcp-project-id>/<artifact-repo-name>/<my-app-name>
```

You can get the `<gcp-region>-docker.pkg.dev/<gcp-project-id>/<artifact-repo-name>` on the repository details page of the repo we just created. Here is an example with all the values replaced with names previously used and my own gcp-project-id.

And you can selected whatever app name you want (we will use `falcon_fron_app`)

```cmd
>>> gcloud auth configure-docker me-central1-docker.pkg.dev
>>> gcloud builds submit --tag=me-central1-docker.pkg.dev/omdena-401512/omdena-falcon-repo/falcon_front_app
```

### 4. Deploy the frontend app on `Cloud Run`

```cmd
>>> gcloud run deploy <my_app_name> --port 8000 --image <gcp-region>-docker.pkg.dev/<project-id>/<artifact-repo-name>/<my-app-name> --allow-unauthenticated --region=<gcp-region> --platform=managed --vpc-connector=<vpc-connector>
```

Let's use the variables defined earlier
```cmd
>>> gcloud run deploy falcon_front_app --port 8000 --image me-central1-docker.pkg.dev/omdena-401512/omdena-falcon-repo/falcon_front_app --allow-unauthenticated --region=me-central1 --platform=managed --vpc-connector=falcon7b-service-connector
```

### 5. Try the app

The output of the previous `gcloud run deploy` command should return the URL to the frontend app. Open this URL and try the app!

## Step 5: Clean GCP

Running all this services is costly... so you better turn them off when you don't need them anymore.

You can delete the project to delete all the ressources at once, or you can stop the GKE cluster (most of the cost comes from this).



## Sources:
- https://github.com/sijohndevoteam/falcon-llm-gke-cluster/tree/main