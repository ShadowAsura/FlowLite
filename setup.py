from setuptools import setup, find_packages

setup(
    name='FlowLite',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.4.0',
        'matplotlib>=3.2.0'
        # add any other dependencies that your project needs
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description='A simplified deep learning library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ShadowAsura/FlowLite',
    author='ShadowAsura',
    author_email='sremlogan@gmail.com',
    license='MIT'
)
