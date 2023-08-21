FROM 141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3
LABEL maintainer="Giuseppe Guarino  <giuseppe@guarinoai.xyz>"

COPY src/requirements.txt ./requirements.txt
RUN mkdir -p /opt/ml/code/
COPY ../src/code/ /opt/ml/code/
RUN pip --no-cache-dir install -r requirements.txt