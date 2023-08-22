FROM 141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3
LABEL maintainer="Giuseppe Guarino  <giuseppe@guarinoai.xyz>"

COPY ./train/code/preprocess.py /preprocess.py

ENTRYPOINT ["python3", "/preprocess.py"]