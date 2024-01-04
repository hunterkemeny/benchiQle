#!/usr/bin/env zsh

# At some point will want to have a for loop so people can run multiple versions of the same compiler, or compare
# more than 2 compilers. 

# TODO: go thru and deal with default values for all of these
echo "Welcome to BenchiQle!"
echo "Enter the first compiler you would like to benchmark (we currently support qiskit and tket): "
read compiler1
echo "Enter the version of the compiler you would like to benchmark (default is latest version): "
read version1
echo "Enter the optimization level for which you would like to run the compiler (default is max optimization level): "
read opt1
echo "Enter the second compiler you would like to benchmark (press enter if you only want to benchmark one compiler): "
read compiler2
if [ -z "$compiler2" ]
then
    echo "No second compiler entered, proceeding with only one compiler."
else
    echo "Benchmarking will be done with $compiler2 as the second compiler."
    echo "Enter the version of the compiler you would like to benchmark (default is latest version): "
    read version2
    echo "Enter the optimization level for which you would like to run the compiler (default is max optimization level): "
    read opt2
fi
echo "Enter the backend you would like to benchmark (default is FakeWashington): "
read backend
echo "Enter the number of times you would like to run each benchmark (default is 1): "
read num_runs

# check and see how many existing venvs there are, and if there is already a venv with these presets (based on a naming convention)
# then dont need to create a new venv, just activate the existing one

# if there is no existing venv, then create a new one with the name of the compiler and version
python3 -m venv venv_
source venv_$version/bin/activate

# need to only install the packages corresponding to what the user wants to benchmark. 
pip install pytket
pip install pytket-qiskit
pip install qiskit==$version
pip install pytest
pip install memory_profiler

python3 runner.py "qiskit" $version > memory_qiskit_$version.txt
deactivate

# should create two venvs, one for each compiler

python3 runner.py "pytket" 0 > memory_pytket_0.txt
