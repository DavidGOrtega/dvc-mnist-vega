virtualenv .env
source .env/bin/activate
pip install -r requirements.txt

mkdir models metrics
touch models/.gitkeep
touch metrics/.gitkeep

dvc init
dvc remote add -d myremote s3://daviddvctest/dvc-mnist-vega

python code/Mnist.py data
dvc add data

dvc run --no-exec \
    -f train.dvc \
    -d code/train.py \
    -d data \
    -o models \
    -M metrics/train.json \
    -M metrics/history.json \
    python code/train.py

dvc run --no-exec \
    -f eval.dvc \
    -d code/eval.py \
    -d data \
    -d models \
    -M metrics/eval.json \
    -M metrics/confusion_matrix.json \
    python code/eval.py


git add --all
git commit -m "setup"

dvc push
git push