from setuptools import setup, find_packages

setup(
    name="pingpong",
    version="0.1",
    packages=find_packages(),  # Automatically finds all packages with __init__.py
    include_package_data=True,  # Includes additional files specified in package_data
    install_requires=[
        # List your dependencies here
        opencv-python-headless,  # OpenCV without GUI dependencies
        opencv-python,  # OpenCV with GUI dependencies
        matplotlib,
        pandas,
        transformers,
        ffmpeg-python,
        ultralytics
    ],
)