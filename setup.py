from setuptools import setup, find_packages

setup(
    name='methdb',
    version='0.0.3',
    packages=find_packages(),
    url='https://github.com/Fu-Yilei/mdb',
    license='MIT',
    author='Yilei Fu',
    description='mdb: population-level DNA methylation analysis toolkit',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
