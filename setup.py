from setuptools import setup

readme = open("README.md", "r")
long_description = readme.read()
readme.close()

setup(
    name='jaccardupy',
    version='0.1',
    author='Justin Boylan-Toomey',
    author_email='justin.boylan-toomey@outlook.com',
    description='Near Duplicate Text Detection Library.',
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
