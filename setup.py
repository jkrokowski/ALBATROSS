from distutils.core import setup
import setuptools
setup(
    name='ALBATROSS',
    version='0.0.1',
    packages=[
        'ALBATROSS'
    ],
    url='https://github.com/jkrokowski/FROOT_BAT',
    author='Joshua Krokowski',
    author_email='jkrokowski@ucsd.edu',
    description="Analysis Library for Beams And Three-Dimensional Representations Of Slender Structures",
    install_requires=[
        'numpy','sparseqr','scipy','fenicsx'
    ]
)