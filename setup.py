from setuptools import setup, find_packages

setup(
    name='cfl_lbm',
    version='0.0.1',
    packages=find_packages(include=['source', 'source.*'])
)

# after setting this up, run `pip install -e .` from task_comp_rnn directory in
# your conda environment