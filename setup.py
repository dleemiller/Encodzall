from setuptools import setup, Extension, find_packages

# Define the extension module
encodzall_module = Extension(
    "encodzall.tokenizer",
    sources=[
        "src/encodzall.c",
        "src/tokenizer.c",
    ],
    extra_compile_args=["-g"],
)

# Define setup parameters
setup(
    name="encodzall",
    version="0.1",
    author="David Lee Miller",
    author_email="dleemiller@gmail.com",
    description="Python C extension for tokenization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dleemiller/encodzall",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
    ],
    packages=find_packages(),
    ext_modules=[encodzall_module],
)
