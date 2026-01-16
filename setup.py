from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """Reads the requirements from a file and returns them as a list."""
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    # ignore -e . and strip whitespace/newlines
    return [req.strip() for req in requirements if req.strip() and not req.startswith('-e .')]

setup(
    name='DiabetesPredictionMLProject',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A machine learning project for diabetes prediction',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)