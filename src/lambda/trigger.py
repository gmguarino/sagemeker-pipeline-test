import json
import boto3


def lambda_handler(event, context):
    sm = boto3.client('sagemaker')
    response = sm.start_pipeline_execution(
        PipelineName='arn:aws:sagemaker:eu-west-1:800075614207:pipeline/TrainingPipelineMLOpsTest',
        PipelineExecutionDisplayName='TrainingPipelineMLOpsTestLambda',
        PipelineExecutionDescription='Triggered by Lambda',
    )

    return {
        'statusCode': 200,
        'body': response
    }
