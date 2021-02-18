# Build Packages for ML Frameworks with KFServing

---

* Ref.
https://github.com/kubeflow/kfserving
https://github.com/GoogleCloudPlatform/ai-platform-samples

## Structure

```bash
${FRAMEWORK_NM}
├── ${FRAMEWORK_NM}.Dockerfile
└── ${FRAMEWORK_NM}server
    ├── Makefile
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    └── ${FRAMEWORK_NM}server
        ├── example_models
        ├── __init__.py
        ├── __main__.py
        ├── model.py
        ├── sklearn_model_repository.py
        ├── test_model.py
        └── test_sklearn_model_repository.py

```


## Architecture

* Project:

```console
${FRAMEWORK_NM}
├── Makefile
├── README.md
├── ${FRAMEWORK_NM}.Dockerfile  <---------------- Dockerfile for Build
├── ${FRAMEWORK_NM}server
│   ├── __init__.py
│   ├── inferencer    <-------------------------- Inference Module
│   └── trainer       <-------------------------- Training Module
├── run_modelserver       <---------------------- Container Entrypoint
└── setup.py
```

* Package `${FRAMEWORK_NM}server`:

```console
${FRAMEWORK_NM}server
├── example_models/0001 <------------------------ Pre-built model files as sample
├── __init__.py
├── inferencer        <-------------------------- Model Inference with API
│   ├── __init__.py
│   ├── __main__.py   <-------------------------- Runnable for Inference API Server
│   ├── model.py      <-------------------------- (inherited from KFserving.Model)
│   ├── model_repository.py  <------------------- Model Loading for Inference
│   ├── test_model.py
│   └── test_model_repository.py
└── trainer           <-------------------------- Model Definition & Training
    ├── 0001
    ├── __init__.py
    ├── __main__.py   <-------------------------- Runnable for Training(Optional)
    ├── config.py       <------------------------ Env. Variables & Config.
    ├── model.py      <-------------------------- Runnable for Training
    ├── postprocessor.py  <---------------------- Postprocessing Module
    ├── preprocessor.py   <---------------------- Preprocessing Module
    └── utils.py         <----------------------- Utility Module
```

---

## Example: Create an model

### `trainer`: Training Module

#### Components

* Model itself
* Pre-processing
* Post-processing

```console
trainer
├── config.py
├── _core.py
├── __init__.py
├── __main__.py
├── model.py
├── postprocessor.py
├── preprocessor.py
└── utils.py
```

#### `trainer.model`: Model Defining

```python
import utils
import config
from .preprocessor import prep_func
from .postprocessor import post_func


class MyModel(object):
    def __init__(self,  dirpath=None, *args, **kwargs):
        if dirpath:
            self.load(dirpath)
        else:
            try:
                self.build(*args, **kwargs)
            except TypeError as e:
                raise e

        self.config = ...

    def preprocess(self, X):
        return prep_func(X)

    def postprocess(self, y_hat):
        return post_func(y_hat)

    def build(self, *args, **kwargs):
        # =========================================== #
        self.model = ...
        # =========================================== #

    def train(self, X, y, *args, **kwargs):
        self.model.fit(X, y)

    def evaluate(self, X, y, *args, **kwargs):
        X = self.preprocess(X)
        y_hat = self.model.evaluate(X, y)
        return self.postprocess(y_hat)

    def predict(self, X, *args, **kwargs):
        X = self.preprocess(X)
        y_hat = self.model.predict(X)
        return self.postprocess(y_hat)

    def save(self, dirpath, *args, **kwargs):
        self.model.save_model(dirpath)

    def load(self, dirpath, *args, **kwargs):
        self.model.load_model(dirpath)

```

### `trainer.__main__`: Runnable for Training

```python
import numpy as np
from sklearnserver.trainer.model import MyModel


train_x, train_y = ...
eval_x, eval_y = ...

model = MyModel(...)
model1.train(train_x, train_y)

eval_res = model.evaluate(eval_x, eval_y)
print(eval_res)
```

Train the model:

```bash
cd ${workspaceFolder}/packages/sklearn
python -m sklearnserver.trainer

...
```

* Output
  * Code: `trainer` module
  * Weights: dumped file from `trainer.model.MyModel.save(...)`

### `inferencer`: Inference Module

```console
inferencer
├── __init__.py
├── __main__.py
├── model.py
├── model_repository.py
├── test_model.py
└── test_model_repository.py
```

#### `inferencer.model`: Model Defining

```python
from sklearnserver.trainer.model import MyModel


# KFModel: tornado based; REST, GRPC support
class SKLearnServingModel(kfserving.KFModel):
    def __init__(self, name: str, model_dir: str):
        ...

    def load(self) -> bool:
        model_path = kfserving.Storage.download(self.model_dir)
        # =========================================== #
        self._model = MyModel(dirpath=model_path)
        # =========================================== #
        self.ready = True
        return self.ready

    def predict(self, request: Dict) -> Dict:
        instances = request["instances"]
        inputs = np.array(instances)
        result = self._model.predict(inputs).tolist()
        return {"predictions": result}

```

* Other preparations:
  * Dependencies: `requirements.txt`
  * `setup.py`
  * Commands, Arguments & Env.: `run_sklearnserver`

## Example: Deploy an trained model

What we've got:

* Code: `sklearnserver` package
  * User-defined: `trainer` module
  * Preset
    * `requirements.txt`
    * `setup.py`
    * `run_sklearnserver`
* Weights: dumped file from `trainer.model.MyModel.save(...)`
  * `example_models/0001/model.joblib`

### Build an Image

#### Dockerfile

```dockerfile
FROM python:3.7-slim

ENV MODEL_NAME
ENV HTTP_PORT="8080"
ENV GRPC_PORT="8081"
ENV MODEL_DIR="/mnt/models"

RUN mkdir -p /workdir
WORKDIR /workdir

COPY sklearnserver sklearnserver
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY README.md README.md
COPY run_sklearnserver /usr/local/bin/run_sklearnserver

RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install -r ./requirements.txt

ENTRYPOINT ["run_sklearnserver"]
```

```bash
cd ${workspaceFolder}/packages/sklearn
docker build -f ./sklearnserver.Dockerfile -t gcr.io/airuntime-templates/sklearnserver:0.1.0 .
```

### Deployment

* Required parameters for Running the image
  * `MODEL_NAME`: used in Request URI

* Input: `input.json`

```json
{"instances": <serialized-input-data> }
```

#### Test in Local

* Deploy a service:

```bash
docker run --rm \
  --mount type=bind,source="$(pwd)/example_models/0001/",target=/mnt/models \
  -e MODEL_NAME=sklearnserver-test \
  -p 8080:8080 \
  sklearnserver:0.1.0
```

* Request a prediction:

```bash
HOST="localhost:8080"
MODEL_NAME="sklearnserver-test"
NAMESPACE="default"
$ curl -X POST "http://${HOST}/v2/models/${MODEL_NAME}/infer" \
    -H "Host: ${MODEL_NAME}.${NAMESPACE}.example.com" \
    -H 'Content-Type: application/json' \
    -d '@./input.json'

{"predictions": [0]}
```

#### Test in Kubernetes

* k8s cluster
* **Model weights file** in a storage can remote access from k8s cluster
  * `gs://airuntime-demo/templates/sklearn/examples/model.joblib`
* **Container image** in a container registry can remote access from k8s cluster
  * `gcr.io/airuntime-templates/sklearnserver:0.1.0`

`sklearnserver.yaml`:

```yml
---
apiVersion: serving.kubeflow.org/v1beta1
kind: InferenceService
metadata:
  name: sklearnserver-test
  namespace: test
spec:
  predictor:
    containers:
    - image: gcr.io/airuntime-templates/sklearnserver:0.1.0
      name: kfserving-container
      env:
        - name: MODEL_NAME
          value: sklearnserver-test
        - name: STORAGE_URI
          value: gs://airuntime-demo/templates/sklearn/examples
      args:
        - --max_buffer_size=104857600
```

* Deploy a service:

```bash
kubectl apply -f sklearnserver.yaml
```

* Request a prediction:

```bash
HOST="<K8S_INGRESS_HOST>"
MODEL_NAME="sklearnserver-test"
NAMESPACE="test"
$ curl -X POST "http://${HOST}/v2/models/${MODEL_NAME}/infer" \
    -H "Host: ${MODEL_NAME}.${NAMESPACE}.example.com" \
    -H 'Content-Type: application/json' \
    -d '@./input.json'

{"predictions": [0]}
```

Done.
