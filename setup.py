#!/usr/bin/env python

import unittest
from setuptools import find_packages, setup


def get_version():
    return "1.5.2post1"


def collect_tests():
    """Sets up unit testing"""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(".", pattern="test_*.py")
    return test_suite


def get_requirements():
    """Collects requirements"""
    with open("requirements.txt", "r") as fp:
        requirements = [line.strip() for line in fp if line.strip()]
        print(requirements)
        return requirements


def main():
    setup(
        name="lasso-python",
        version=get_version(),
        description="LASSO Python Library",
        author="open-lasso-python community",
        url="https://github.com/open-lasso-python/lasso-python",
        license="BSD 3-Clause",
        packages=find_packages(),
        install_requires=get_requirements(),
        package_data={
            "": [
                "*.png",
                "*.html",
                "*.js",
                "*.so",
                "*.so.5",
                "*.dll",
                "*.txt",
                "*.css",
            ],
        },
        test_suite="setup.collect_tests",
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
