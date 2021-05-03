from setuptools import setup, find_packages
from distutils.core import setup
setup(name='eigenwell',
      version='1.0',
      py_modules=['eigenwell'],
      install_requires=['numpy', 'scipy', 'matplotlib'],
      author_email='nzz2102@stanford.edu.com'
      )

# setup(
#    name='eigenwell',
#    version='1.0',
#    description='OO eigensolves',
#    author='Nathan Zhao',
#    author_email='nzz2102@stanford.edu.com',
#    packages=['foo'],  #same as name
#    install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
# )
