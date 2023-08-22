FROM 141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3
LABEL maintainer="Giuseppe Guarino  <giuseppe@guarinoai.xyz>"

COPY ./train/requirements/train_requirements.txt ./train_requirements.txt
RUN mkdir -p /opt/ml/code/
COPY ./train/code/train.py /opt/ml/code/train.py
RUN pip --no-cache-dir install -r train_requirements.txt

ENV SAGEMAKER_PROGRAM train.py
