from setuptools import setup, find_packages
import platform

setup(
    name='flash_swin_attn',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'einops',
        'triton-windows' if platform.system() == 'Windows' else 'triton'
    ],

    entry_points={
        'console_scripts': [
            # Add command line scripts here
        ],
    },
    author='Zhendong Zhang',
    author_email='zhd.zhang.ai@gmail.com',
    description='Speedup the attention computation of Swin Transformer',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/zzd1992/FlashSwinAttention',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)