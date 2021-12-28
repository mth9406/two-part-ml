from setuptools import find_packages, setup

setup(
    name= 'two_part_ml'
    version= '0.0.1',
    description= 'Implemenation of two-part-ml',
    author= 'SUNGWOO HUR',
    author_email= 'hursungwoo@postech.ac.kr',
    url= 'https://github.com/mth9406/two-part-ml.git',
    install_requires= ['catboost>=1.0.3', 'numpy>=1.16.0', 'sklearn>=1.0.1' ],
    packages = find_packages(),
    zip_safe = False
)