## Anomaly Web Dashboard

How to run server:

``python app.py``

How to train: 

``curl -X POST -F "file=@$(pwd)/dataset/cubes.zip" http://localhost:8000/train``

How to test:

``curl -X POST -F "file=@$(pwd)/dataset/cubes/abnormal/input_20230210134059.jpg" http://localhost:8000/predict``