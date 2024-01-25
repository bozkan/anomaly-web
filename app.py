from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from anomalib.models import Padim
from anomalib.data.folder import Folder
from anomalib.data.task_type import TaskType
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from anomalib.deploy import OpenVINOInferencer
from anomalib.data.utils import read_image
from threading import Thread
import shutil
import os
import zipfile

app = Flask(__name__)
# Configure CORS
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})

def train_anomaly_model(dataset_path):
    # Prepare Dataset
    datamodule = Folder(
        root=Path(dataset_path),
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
        ModelCheckpoint(
            mode="max",
            monitor="image_AUROC",
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

    # Save model for inference
    model_path = Path.cwd() / "model.ckpt"
    trainer.save_checkpoint(model_path)

@app.route('/train', methods=['POST'])
def train():
    file = request.files['file']
    dataset_path = Path.cwd() / "dataset"
    dataset_zip_path = dataset_path / 'dataset.zip'
    
    # Save and Extract Dataset
    file.save(dataset_zip_path)
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    os.remove(dataset_zip_path)

    # Start training in a separate thread
    thread = Thread(target=train_anomaly_model, args=(dataset_path,))
    thread.start()
    return jsonify({'message': 'Training started'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image_path = Path.cwd() / "temp_image.jpg"
    file.save(image_path)

    # Load model and perform inference
    model_path = Path.cwd() / "model.ckpt"
    inferencer = OpenVINOInferencer(
        path=model_path,
        device="CPU",
    )
    image = read_image(str(image_path))
    predictions = inferencer.predict(image=image)
    
    # Assume predictions contain anomaly score (customize as needed)
    anomaly_score = predictions.get('score', 0.5)

    return jsonify({'anomaly_rate': anomaly_score}), 200

if __name__ == '__main__':
    app.run(debug=True)
