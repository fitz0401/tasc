from setuptools import setup, find_packages

setup(
    name='tasc',
    version='1.0',
    packages=find_packages(),
    description='Official implementation of TASC: Task-Aware Shared Control for Teleoperated Manipulation',
    url='git@github.com:fitz0401/tasc.git',
    author='ze fu',
    author_email='ze.fu@kuleuven.be',
    license='MIT',
    install_requires=[
        'typing',
        'typing_extensions',
    ],
    zip_safe=False
)