from setuptools import find_packages, setup
# DevSkiller python environment has a low version of sqlit and its not able to install the newer version. this is required for Chromadb.
# the test works fine and pass all 4 tests (with 2 warnings) on my local machine as per instructutions, I will send more details in the email confirmation.

setup(
    name="my-rag",
    version="1.0.0",
    author="Hesham Fahim",
    author_email="hesham.fahim@gmail.com",
    packages=find_packages(),
    test_suite="test",
    install_requires=[
        "wheel",
        "pandas==2.2.*",
        "transformers==4.43.*",
        "ragas==0.1.*",
        "pypdf==4.3.*",
        "python-dotenv==1.0.*",
        "langchain==0.2.*",
        "chromadb",
    ],
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
        "pytest-timeout",
    ],
    extras_require={
        'test': [
            "pytest",
            "pytest-timeout",
        ],
    },
)