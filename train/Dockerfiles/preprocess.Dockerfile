FROM 141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3
LABEL maintainer="Giuseppe Guarino  <giuseppe@guarinoai.xyz>"

RUN mkdir -p /opt/ml/code/
COPY ./train/code/preprocess.py /opt/ml/code/preprocess.py

ENTRYPOINT ["python3", "/opt/ml/code/preprocess.py"]