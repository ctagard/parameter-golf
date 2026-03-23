from setuptools import setup, find_packages

setup(
    name="looped-gpt-kernels",
    version="0.1.0",
    description="Fused CUDA kernels for LoopedGPT mHC-lite architecture",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "triton": ["triton>=2.1.0"],
    },
)
