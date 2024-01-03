#!/usr/bin/env zsh

versions=(0.45.0)

for version in "${versions[@]}"
do
    python3 -m venv venv_$version
    source venv_$version/bin/activate
    pip install pytket
    pip install pytket-qiskit
    pip install qiskit==$version
    pip install pytest
    pip install memory_profiler
    python3 runner.py "qiskit" $version > memory_qiskit_$version.txt
    deactivate
done

python3 runner.py "pytket" 0 > memory_pytket_0.txt
