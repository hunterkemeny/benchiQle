#!/usr/bin/env zsh

# At some point will want to have a for loop so people can run multiple versions of the same compiler, or compare
# more than 2 compilers. 

# TODO: go thru and deal with default values for all of these
echo "Welcome to BenchiQle!"
echo "Enter the first compiler you would like to benchmark (we currently support qiskit and tket): "
read compiler1
echo "Enter the version of the compiler you would like to benchmark: "
read version1
echo "Enter the optimization level for which you would like to run the compiler: "
read opt1
echo "Enter the second compiler you would like to benchmark (press enter if you only want to benchmark one compiler): "
read compiler2
if [ -z "$compiler2" ]
then
    echo "No second compiler entered, proceeding with only one compiler."
else
    echo "Benchmarking will be done with $compiler2 as the second compiler."
    echo "Enter the version of the compiler you would like to benchmark: "
    read version2
    echo "Enter the optimization level for which you would like to run the compiler: "
    read opt2
fi
# TODO: want to allow for users to compare compilers with different backends? And with different runs for each compiler?
echo "Enter the backend you would like to benchmark (default is IBM FakeWashington): "
read backend
echo "Enter the number of times you would like to run each benchmark (default is 1): "
read num_runs
if [ -z "$num_runs" ]
then
    num_runs=1
fi

# TODO: make this modular so there is just a call to a function that does this for compiler 1 and 2 (so code doesnt repeat?)
# Naming convention: venv_compilerName_versionNumber
venv_name="venv_${compiler1}_${version1}"
cd virtual_environments
if [ -d "$venv_name" ]
then
    echo "Starting up virtual environment $venv_name."
    source $venv_name/bin/activate
else
    echo "Virtual environment $venv_name does not yet exist on your system."
    echo "Creating virtual environment $venv_name."
    python3 -m venv $venv_name
    source $venv_name/bin/activate
    pip install memory_profiler
    pip install pytest
    pip install numpy
    # TODO: installing tket because it is being imported through utils; possibly remove this dependency or restructure
    # TODO: may not need pytket (just pytket qiskit); also may not need pytest, also may need to rearrange 
    # because we may not be able to run runner.py without installing pytket; also need to insert version for pytket
    # TODO: may want to check tket compiler first, so then we can download the version that the user chose
    pip install pytket
    pip install pytket-qiskit
    if [ "$compiler1" = "qiskit" ]
    then
        pip install qiskit==$version1
    elif [ "$compiler1" = "pytket" ]
    then
        echo "tket already installed"
    else
        # TODO: this check should come earlier
        echo "Compiler $compiler1 is not supported."
        exit 1
    fi
fi
# TODO: should I suppress the pip install outputs?
cd ..
python3 runner.py $compiler1 $version1 $opt1 $backend $num_runs > memory_${compiler1}_$version1.txt
deactivate

if [ -z "$compiler2" ]
then
    echo "No second compiler entered, exiting."
    exit 0
fi

venv_name="venv_${compiler2}_${version2}"
if [ -d "$venv_name" ]
then
    echo "Starting up virtual environment $venv_name."
    source $venv_name/bin/activate
else
    echo "Virtual environment $venv_name does not yet exist on your system."
    echo "Creating virtual environment $venv_name."
    python3 -m venv $venv_name
    source $venv_name/bin/activate
    pip install memory_profiler
    pip install pytest
    if [ "$compiler2" == "qiskit" ]
    then
        pip install qiskit==$version2
    elif [ "$compiler2" == "pytket" ]
    then
        pip install pytket
        pip install pytket-qiskit
    else
        echo "Compiler $compiler2 is not supported."
        exit 1
    fi
fi

python3 runner.py $compiler2 $version2 $opt2 $backend $num_runs > memory_${compiler2}_$version2.txt
deactivate