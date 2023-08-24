from datetime import datetime

import sagemaker
import argparse
import json
import os

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput

from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
from sagemaker.workflow.properties import PropertyFile


def get_sagemaker_session():
    """
    Gets sagemaker session, bucket.
    """
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()

    return sagemaker_session, bucket


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model id REQUIRED 
    parser.add_argument("--model-id", type=str, required=True)

    # preprocess job
    parser.add_argument("--preprocess-job-name", type=str, default="mlops-test-nb-pipeline-preprocess")
    parser.add_argument("--preprocess-step-name", type=str, default="MLOpsTestProcessing")
    parser.add_argument("--preprocess-entry-point", type=str, default=os.path.join("../../train/code/preprocess.py"))
    
    # train job
    parser.add_argument("--train-step-name", type=str, default="mlops-pipeline-test-model")
    parser.add_argument("--train-entry-point", type=str, default="train.py")
    parser.add_argument("--train-source-dir", type=str, default="../../train/code/")

    # eval job
    parser.add_argument("--evaluate-job-name", type=str, default="mlops-test-nb-pipeline-evaluate")
    parser.add_argument("--evaluate-entry-point", type=str, default="../../train/code/evaluate.py")
    parser.add_argument("--evaluate-step-name", type=str, default="EvaluatePerformance")

    # s3 prefix
    parser.add_argument("--prefix", type=str, default="projects/mlops")
    
    # naming
    parser.add_argument("--base-job-prefix", type=str, default="mlops-test")
    parser.add_argument("--model-package-group-name", type=str, default="MLOpsTestModel")
    parser.add_argument("--pipeline-name", type=str, default="TrainingPipelineMLOpsTest")
    parser.add_argument("--role", type=str, default="arn:aws:iam::436376758376:role/service-role/SageMaker-MLOpsEngineer1")

    
    
    args = parser.parse_args()

    # logs = {}
    # model_id_prefix = "sklearn-dummy"
    # date_str = datetime.now().strftime("%Y-%m-%d")
    # args.model_id = f'{model_id_prefix}-{date_str}-' + secrets.token_hex(nbytes=16)
    # print(args.model_id)
    now = datetime.now()

    # parameters for pipeline execution
    processing_instance_count = 1
    evaluation_instance_count = 1
    processing_instance_type = "ml.m5.large"
    training_instance_type = "ml.m5.large"
    evaluation_instance_type = "ml.m5.large"
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    sagemaker_session, bucket = get_sagemaker_session()

    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        volume_size_in_gb=5,
        base_job_name=args.preprocess_job_name,
        role=args.role
    )

    processing_step = ProcessingStep(
        name=args.preprocess_step_name,
        processor=sklearn_processor,
        outputs=[
            ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train",
                            destination=os.path.join(f"s3://{bucket}", args.prefix, "challenger", 
                                                    now.strftime("%Y/%m/%d"), args.model_id, "data", "train")),
            ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test",
                            destination=os.path.join(f"s3://{bucket}", args.prefix, "challenger", 
                                                    now.strftime("%Y/%m/%d"), args.model_id, "data", "test")),
        ],
        code=args.preprocess_entry_point
    )

    sklearn = SKLearn(
        entry_point=args.train_entry_point,
        source_dir=args.train_source_dir,
        framework_version="1.2-1",
        instance_type=training_instance_type,
        instance_count=1,
        role=args.role,
        sagemaker_session=sagemaker_session,
        output_path=f"s3://{bucket}/{args.prefix}/training/output",
        code_location=f"s3://{bucket}/{args.prefix}/training/code"
    )

    step_train = TrainingStep(
        name=args.train_step_name,
        estimator=sklearn,
        inputs={
            "train": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                content_type="text/csv",
            )
        }
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    evaluation = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=evaluation_instance_type,
        instance_count=evaluation_instance_count,
        volume_size_in_gb=5,
        base_job_name=args.args.evaluate_job_name,
        role=args.role
    )
    step_evaluate = ProcessingStep(
        name=args.evaluate_step_name,
        processor=evaluation,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", destination=os.path.join(f"s3://{bucket}", args.prefix, "challenger", 
                                                    now.strftime("%Y/%m/%d"), args.model_id, "data", "evaluation_report")),
        ],
        property_files=[evaluation_report],
        code=args.evaluate_entry_point
    )

    pipeline = Pipeline(
        name=args.pipeline_name,
        steps=[processing_step, step_train, step_evaluate]
    )
    json.loads(pipeline.definition())
    pipeline.upsert(role_arn=args.role)
