import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='bebi103ahw93_utilities',
    version='0.0.1',
    author='Daniel Pollak',
    author_email='dpollak@caltech.edu',
    description='bebi103a group 13 is learning how to make packages.',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    install_requires=["numpy","pandas", "bokeh>=1.4.0"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)