from distutils.core import setup
# import setuptools
# from setuptools import setup,find_packages
# with open("README.md","r") as f:
#     long_description = f.read()
setup(
    name='ALBATROSS',
    version='0.1.0',
    # package_dir={"":"ALBATROSS"},
    packages=['ALBATROSS'],
    # packages=find_packages(where='ALBATROSS'),
    url='https://github.com/jkrokowski/ALBATROSS',
    author='Joshua Krokowski',
    author_email='jkrokowski@ucsd.edu',
    description="Analysis Library for Beams And Three-Dimensional Representations Of Slender Structures",
    # long_description=long_description,
    long_description_content_type = "text/markdown",
    # install_requires=[
    #     'numpy',
    # ]
    install_requires=[
        'numpy',
        'sparseqr',
        'scipy',
        'pytest',
        'myst-nb',
        'sphinx==5.3.0',
        'sphinx_rtd_theme',
        'sphinx-copybutton',
        'sphinx-autoapi==2.1.0',
        'astroid==2.15.5',
        'numpydoc',
        'gitpython',
        'sphinxcontrib-collections @ git+https://github.com/anugrahjo/sphinx-collections.git', # 'sphinx-collections',
        'sphinxcontrib-bibtex',
        'setuptools',
        'wheel',
        'twine',
    ],
)