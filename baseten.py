from truss.base import truss_config
from truss_train import definitions

runtime = definitions.Runtime(
    start_commands=[
        "sleep infinity",  # fill in with your start command
    ],
    environment_variables={
        "HF_TOKEN": definitions.SecretReference(
            name="hf_access_token"
        ),  # set envvars, reference secrets in your workspace
        "aws_secret_access_key": definitions.SecretReference(name="aws_secret_access_key"),
        "aws_access_key_id": definitions.SecretReference(name="aws_access_key_id"),
        "aws_region": definitions.SecretReference(name="aws_region"),
        "aws_data_bucket_name": definitions.SecretReference(name="aws_data_bucket_name"),
        "wandb_api_key": definitions.SecretReference(name="wandb_api_key"),
        "AUTH_COOKIE_SECRET": definitions.SecretReference(name="neo_encryption_key"),
        "baseten_api_key": definitions.SecretReference(name="baseten_api_key"),
    },
)

training_job = definitions.TrainingJob(
    compute=definitions.Compute(
        accelerator=truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator.H200,
            count=2,
        ),
        cpu_count=4,
        memory="32Gi",
    ),
    runtime=runtime,
    # this can be private or public images
    image=definitions.Image(base_image="pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel"),
    workspace=definitions.Workspace(
        workspace_root="./",
        exclude_dirs=[],
        external_dirs=["/Users/allanbishop/Projects/DDPRPO"],
    ),
    interactive_session=definitions.InteractiveSession(
        trigger=definitions.InteractiveSessionTrigger.ON_STARTUP,
        # can also be GITHUB
        auth_provider=definitions.InteractiveSessionAuthProvider.GITHUB,
        # can also be CURSOR, but must follow setup instructions here
        # https://www.notion.so/ml-infra/Cursor-RSSH-Setup-2c691d247273807d92b6ee3a39eefabf?source=copy_link
        session_provider=definitions.InteractiveSessionProvider.CURSOR,
        # Session will be kept alive for 5 days
        timeout_minutes=60 * 24 * 7,
    ),
)

_ = definitions.TrainingProject(name="ask-latent-grpo", job=training_job)
