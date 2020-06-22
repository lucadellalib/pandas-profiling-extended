from os.path import dirname, join, realpath

from setuptools import find_packages, setup


ROOT_DIR = dirname(realpath(__file__))

VERSION = {}

with open(join(ROOT_DIR, "pandas_profiling", "version.py"), "r") as version_file:
    exec(version_file.read(), VERSION)

with open(join(ROOT_DIR, "README.md"), encoding="utf-8") as f:
    README = f.read()

with open(join(ROOT_DIR, "requirements.txt"), encoding="utf-8") as f:
    REQUIREMENTS = f.read().splitlines()

setup(
    name="pandas-profiling-extended",
    version=VERSION["VERSION"],
    author="Luca Della Libera",
    author_email="luca310795@gmail.com",
    packages=find_packages(),
    license="MIT",
    description="Generate profile report for pandas DataFrame",
    python_requires=">=3.6",
    install_requires=REQUIREMENTS,
    extras_require={
        "notebook": ["jupyter-client>=6.0.0", "jupyter-core>=4.6.3"],
        "app": ["pyqt5>=5.14.1"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Environment :: Console",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering",
        "Framework :: IPython",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="pandas data-science data-analysis python jupyter ipython",
    long_description=README,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "pandas_profiling = pandas_profiling.controller.console:main"
        ]
    },
    options={"bdist_wheel": {"universal": True}},
    zip_safe=True
)
