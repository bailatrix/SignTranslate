from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = [
    'matplotlib',
    'opencv-python',
    'pandas',
    'seaborn',
    'requests >= 2.18.0',
    'sklearn',
    'tqdm',
    'scikit-image',
    'google-resumable-media[requests]'
]

EXTRAS_REQUIRE = {
    'requests': [
        'requests >= 2.18.0, < 3.0.0dev',
    ],
}

setup(
    name="signtraining",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRAS_REQUIRE,
    packages=find_packages(),
    include_package_data=True,
    description='sign training'
)
