from setuptools import setup, find_packages

setup(
    name="grokking-beyond-addition",
    version="3.0.0",   # FIX: was 2.0.0 — bumped to match project version
    author="Mani Pal",
    author_email="justbytecode@users.noreply.github.com",
    description="Circuit-level analysis of algebraic learning in transformers",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/justbytecode/grokking-beyond-addition",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformer_lens>=1.14.0",  # must pair with numpy==1.26.4: wheels compiled against numpy 1.x
        "einops>=0.7.0",
        "plotly>=5.18.0",
        "matplotlib>=3.8.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0,<2.0.0",     # FIX BUG-11: cap <2.0.0; TL wheels use numpy 1.x ABI
        "tqdm>=4.66.0",
        "pandas>=2.1.0",
        "kaleido==0.2.1",           # pin exactly: kaleido 0.3+ breaks Plotly static export
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
