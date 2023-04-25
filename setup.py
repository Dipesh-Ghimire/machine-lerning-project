from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
   """
      This function takes a file path as input and returns a list of requirements extracted from the given file.
      Each requirement is a string without the "\n" character and without the '-e .' substring (if present in the file).
      
      :param file_path: A string representing the path of the file containing requirements.
      :type file_path: str
      
      :return: A list of requirements extracted from the file.
      :rtype: List[str]
   """

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
