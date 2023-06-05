from distutils.core import setup
import setuptools
setup(
    name='FROOT_BAT',
    version='0.1',
    packages=[
        'FROOT_BAT'
    ],
    url='https://github.com/jkrokowski/FROOT_BAT',
    author='Joshua Krokowski',
    author_email='jkrokowski@ucsd.edu',
    description="FEniCSx Represenation Of slender Objects for Three-Dimensional Beam Analysis Tasks",
    install_requires=[
        'numpy'
    ]
)