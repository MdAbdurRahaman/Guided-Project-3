import setuptools

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "End to End MLOps Data Science Project Implementation"

__version__ = "0.0.0"

REPO_NAME = "End-to-End-MLOps-Wine-Quality-Project"
AUTHOR_USER_NAME = "Abdur R"
SRC_REPO = "mlProject"
AUTHOR_EMAIL = "developer@example.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Python package for Wine Quality Prediction MLOps project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
