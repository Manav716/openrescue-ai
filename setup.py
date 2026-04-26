from setuptools import find_packages, setup


setup(
    name="openrescue-ai",
    version="1.0.0",
    description="OpenEnv-compatible neuro-symbolic multi-agent disaster response environment.",
    packages=find_packages(include=["env", "env.*"]),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29",
        "numpy>=1.24",
        "stable-baselines3[extra]>=2.3",
        "transformers>=4.40",
        "torch>=2.2",
        "pygame>=2.5",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.0",
        "matplotlib>=3.8",
        "openenv-core",
    ],
)
