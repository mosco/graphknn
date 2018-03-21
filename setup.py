from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='graphknn',
      version='0.2',
      description='For every vertex in a graph, this code efficiently finds its K nearest vertices from a particular subset of "special" vertices.',
      url='https://github.com/mosco/graphknn',
      author='Amit Moscovich',
      author_email='moscovich@gmail.com',
      license='MIT',
      keywords=['graph knn dijkstra shortest path'],
      py_modules=['graphknn'],
      install_requires=['SciPy', 'HeapDict'],
      long_description=read('README.md'),
      classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
      ],
      zip_safe=True)
