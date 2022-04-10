import setuptools

# python setup.py install

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='portfolio_optimization',
    version='0.0.1',
    author='Hugo Delatte',
    author_email='delatte.hugo@gmail.com',
    description='Portfolio Optimisation: from Markowitz to Genetic Algorithm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)