from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT= "-e ."
def get_requirements(filepath: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements= []

    with open(filepath) as file_obj:
        requirements= file_obj.readlines()
        requirements= [i.replace("\n", " ") for i in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name= "ML_Pipeline_Project",
    version= "0.0.1",
    description= "Machine Learning Pipeline Project",
    author= "Shivang Shritu",
    author_email= "shivang.cse.nitnagaland@gmail.com",
    packages= find_packages(),
    install_requires= get_requirements("requirements.txt")
)