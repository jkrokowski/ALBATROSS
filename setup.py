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
    install_requires=[
        'numpy',
        'scipy',
        # 'sparseqr',
        # 'fenics-dolfinx',
        # 'meshio',
        # 'pyvista',
        # 'sparseqr @ git+https://github.com/yig/PySPQR.git',
        'gmsh'
    ]
)