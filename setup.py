from setuptools import setup

readme = open("README.md", "r")
long_description = readme.read()
readme.close()

setup(
    name='jaccardupy',
    version='0.2',
    author='Justin Boylan-Toomey',
    author_email='justin.boylan-toomey@outlook.com',
    description='Jaccardupy is a Python library for detecting near duplicate texts'
                ' in a corpus using Locality Sensitive Hashing.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/justinnbt/Jaccardupy',
    packages=['jaccardupy'],
    install_requires=['numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
