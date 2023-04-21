from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
   # this function returns list of requirements

   requirements = []
   with open(file_path) as file_obj:
      requirements = file_obj.readlines()
      # Remove the "\n" in requirements
      requirements = [req.replace("\n","") for req in requirements]

      # Remove '-e .' in requirements which came from last line of requirements.txt
      if HYPEN_E_DOT in  requirements:
         requirements.remove(HYPEN_E_DOT)
   return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Dipesh Ghimire',
    author_email='dipeshghimire.dg@gmail.com',
    description='My Main Python package',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
