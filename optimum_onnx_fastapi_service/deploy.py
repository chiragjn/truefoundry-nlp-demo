import argparse
import logging

from servicefoundry import (
    Service,
    PythonBuild,
    Resources,
    HttpProbe,
    Build,
    HealthProbe,
)

logging.basicConfig(level=logging.INFO)


def deploy(workspace_fqn: str):
    service = Service(
        name="sum-ort-fastapi",
        image=Build(
            build_spec=PythonBuild(
                python_version=3.9,
                command="bash run.sh",
                requirements_path="requirements.txt",
            )
        ),
        ports=[{"port": 8000}],
        resources=Resources(
            cpu_request=4,
            cpu_limit=5,
            memory_request=5000,
            memory_limit=10000,
            instance_family=["c6i"]
        ),
    )

    service.deploy(workspace_fqn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-fqn", required=True, type=str)
    args = parser.parse_args()

    deploy(workspace_fqn=args.workspace_fqn)
