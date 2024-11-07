from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ebook2audiobook",  # Replace with your desired package name
    version="0.0.8",
    author="Andrew Phillip Thomasson",
    author_email="drew.thomasson100@gmail.com",
    description="Convert eBooks to Audiobooks using a Text-to-Speech model with optional Gradio interface.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DrewThomasson/ebook2audiobookXTTS",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "coqui-tts==0.24.2",
        "pydub",
        "nltk",
        "beautifulsoup4",
        "ebooklib",
        "tqdm",
        "gradio==4.44.1"
    ],
    entry_points={
        'console_scripts': [
            'ebook2audiobook=ebook2audiobook.app:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Update if using a different license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7, <3.13',  # Specify your supported Python versions
)

