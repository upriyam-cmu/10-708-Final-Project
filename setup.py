from distutils.core import setup

setup(
    name="edge-rec",
    version="0.0.0",
    install_requires=[
        'accelerate',
        'einops',
        'ema-pytorch>=0.4.2',
        'numpy',
        'pillow',
        'pytorch-fid',
        'torch',
        'torchvision',
        'tqdm'
    ],
    author="",
    author_email="",
    url="none",
    packages=["edge_rec"],
)

