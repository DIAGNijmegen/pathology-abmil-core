from setuptools import setup, find_packages

setup(
    name="clam_latest",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "timm==0.9.8",
        "torch",
        "torchvision",
        "h5py",
        "pandas",
        "PyYAML",
        "opencv-python",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "tqdm",
        "openslide-python",
        "tensorboardX",
        "typeguard"
    ],
    include_package_data=True,
)
