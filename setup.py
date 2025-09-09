from setuptools import setup, find_packages

e_dot="-e ."
def get_requirements(file_path:str)->list:
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
    
    if e_dot in requirements:
        requirements.remove(e_dot)
    
    return requirements

setup(
    name='my_package',
    version='0.0.1',    
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn'],
    author='Abhishek',
    install_requires=get_requirements('requirements.txt')
)     
