from setuptools import setup, find_packages

setup(
    name="countdown",  # Name of your package
    packages=["countdown", "ops", "prep_data"],  # Include all packages
    package_dir={"": "."},  # Base directory is current directory
)