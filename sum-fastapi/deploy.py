import argparse
import logging
from typing import Optional

from servicefoundry import (
    Service,
    PythonBuild,
    Resources,
    Build,
)

logging.basicConfig(level=logging.INFO)


def deploy(workspace_fqn: str, num_threads: Optional[int]):
    service = Service(
        name="sum-fastapi",
        image=Build(
            build_spec=PythonBuild(
                python_version=3.9,
                command="gunicorn app:app --timeout 120 --workers 3 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --threads 1 --access-logfile -",
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
        env={
            "NUM_THREADS": num_threads if num_threads else '',
            # "API_TOKEN": "..."
        }
    )

    service.deploy(workspace_fqn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-fqn", required=True, type=str)
    parser.add_argument("--num-threads", type=int, default=None)
    args = parser.parse_args()

    deploy(workspace_fqn=args.workspace_fqn, num_threads=args.num_threads)
