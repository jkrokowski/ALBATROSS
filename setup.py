from distutils.core import setup
import setuptools
setup(
    name='FRuIT_BAT',
    version='0.1',
    packages=[
        'FRuIT_BAT'
    ],
    url='https://github.com/jkrokowski/FRuIT_BAT',
    author='Josh Krokowski',
    author_email='jkrokowski@ucsd.edu',
    description="FEniCSx Represenation of Intervals for Three-Dimensional Beam Analysis Toolkit",
    install_requires=[
        'numpy'
    ]
)