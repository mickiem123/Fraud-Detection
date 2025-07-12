from setuptools import setup, find_packages

setup(
    name="fraud_detection",
    version="0.1.0",
    author="Huan",
    author_email="your.email@example.com",
    description="A machine learning project for fraud detection.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ],
    python_requires=">=3.7",
)

