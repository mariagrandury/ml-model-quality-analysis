from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()


setup(
    name='ml-model-quality-analysis',
    version='0.0.1',
    author='María Grandury',
    author_email='mariagrandury@gmail.com',
    description='A package to perform quality analyses for Machine Learning models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mariagrandury/ml-model-quality-analysis',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
)