import argparse
import logging

from servicefoundry import Build, Job, PythonBuild, Resources
logging.basicConfig(level=logging.INFO)

def deploy(workspace_fqn: str):
    for i, instance_family in enumerate(["t3", "m6a", "c6i"]):
        for j, (cpu_limit, memory_limit) in enumerate([(2, 4000), (4, 6000)]):
            print(f"-> Deploying CPU {cpu_limit} MEM {memory_limit} INSTANCE {instance_family}")
            job = Job(
                name=f"tmp-b-{i}-{j}-{instance_family}",
                image=Build(
                    build_spec=PythonBuild(
                        python_version="3.9",
                        command="python run.py",
                    )
                ),
                env={
                    "MACHINE_TYPE": instance_family,
                    "CPU_LIMIT": str(cpu_limit),
                    "MEM_LIMIT": str(memory_limit),
                    "TFY_API_KEY": "tfy-secret://user-truefoundry:nlp-demo-benchmarking:TFY_API_KEY",
                },
                resources=Resources(
                    cpu_request=1, 
                    cpu_limit=cpu_limit, 
                    memory_request=2000, 
                    memory_limit=memory_limit,
                    instance_family=[instance_family],
                ),
            )
            job.deploy(workspace_fqn=workspace_fqn)
            # input("Continue?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-fqn", required=True, type=str)
    args = parser.parse_args()
    deploy(workspace_fqn=args.workspace_fqn)
