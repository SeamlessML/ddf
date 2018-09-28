from setuptools import setup
import re

with open('ddf/__init__.py', 'r') as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(),
        re.MULTILINE
        ).group(1)


setup(
        name='ddf',
        version=version,
        packages=['ddf'],
        install_requires=[
            'numpy',
            'pandas<0.22.1'
            ],
        )
