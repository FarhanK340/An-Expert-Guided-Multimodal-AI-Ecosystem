from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="expert-mome-fyp",
    version="0.1.0",
    author="Farhan",
    author_email="fkashif.bese22seecs@seecs.edu.pk",
    description="Mixture of Modality Experts (MoME+) for Medical Image Segmentation with Continual Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FarhanK340/An-Expert-Guided-Multimodal-AI-Ecosystem",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
        ],
        "deploy": [
            "gunicorn>=21.2.0",
            "psycopg2-binary>=2.9.0",
            "whitenoise>=6.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mome-train=training.trainer:main",
            "mome-infer=inference.inference_engine:main",
            "mome-preprocess=preprocessing.data_preprocessing:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "*.json"],
    },
)

