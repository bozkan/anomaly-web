import zipfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from anomalib.models import Padim
from anomalib.data.folder import Folder
from anomalib.data.task_type import TaskType
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from anomalib.deploy import OpenVINOInferencer
from anomalib.data.utils import read_image
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.callbacks.export import ExportCallback, ExportMode
from threading import Thread
from pathlib import Path
import shutil
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def train_anomaly_model(dataset_path: str):

    # Prepare Dataset
    datamodule =
    Folder(
        root=dataset_path,
        normal_dir="normal",
        abnormal_dir="abnormal",
        normal_split_ratio=0.2,
        image_size=(256, 256),
        train_batch_size=32,
        eval_batch_size=32,
        task=TaskType.CLASSIFICATION,
    )
    datamodule.setup()
    datamodule.prepare_data()

    # Initialize Model
    model = Padim(
        input_size=(256, 256),
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )

    # Prepare Callbacks
    callbacks = [
        MetricsConfigurationCallback(
            task=TaskType.CLASSIFICATION,
            image_metrics=["AUROC"],
        ),
        ModelCheckpoint(
            mode="max",
            monitor="image_AUROC",
        ),
        PostProcessingConfigurationCallback(
            normalization_method=NormalizationMethod.MIN_MAX,
            threshold_method=ThresholdMethod.ADAPTIVE,
        ),
        MinMaxNormalizationCallback(),
        ExportCallback(
            input_size=(256, 256),
            dirpath=str(Path.cwd()),
            filename="model",
            export_mode=ExportMode.OPENVINO,
        ),
    ]

    # Create Trainer and Train the Model
    trainer = Trainer(
        callbacks=callbacks,
        accelerator="auto",
        auto_scale_batch_size=False,
        max_epochs=1,
        check_val_every_n_epoch=1,
        devices=1,
        gpus=None,
        num_sanity_val_steps=0,
        val_check_interval=1.0,
    )
    trainer.fit(model=model, datamodule=datamodule)


@app.post('/train')
async def train(file: UploadFile = File(...)):
  
    try:
        data_folder_path = Path.cwd() / "dataset"
        dataset_zip_path = data_folder_path / file.filename
        dataset_path = data_folder_path / file.filename.replace(".zip", "")
        
        # Create the dataset folder if it doesn't exist
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        # Save and Extract Dataset
        with open(dataset_zip_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_folder_path)
        os.remove(dataset_zip_path)

        # Start training in a separate thread
        thread = Thread(target=train_anomaly_model, args=(dataset_path,))
        thread.start()

        return JSONResponse(content={'message': 'Training started'})
    except Exception as e:  
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Get the original filename of the uploaded image
    image_filename = file.filename

    # Use the original filename as the image path
    image_path = Path.cwd() / image_filename

    with open(image_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load model and perform inference
    openvino_model_path = Path.cwd() / "weights" / "openvino" / "model.bin"
    metadata_path = Path.cwd() / "weights" / "openvino" / "metadata.json"

    inferencer = OpenVINOInferencer(
        path=openvino_model_path,
        metadata=metadata_path,
        device="CPU",
    )

    image = read_image(str(image_path))
    predictions = inferencer.predict(image=image)

    # Access the anomaly score directly
    anomaly_score = predictions.anomaly_score

    return JSONResponse(content={'anomaly_rate': anomaly_score})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
