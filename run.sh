#!/usr/bin/env zsh

versions=(0.35.0 0.37.0 0.39.0 0.41.0 0.43.0 0.44.0)

for version in "${versions[@]}"
do
    python3 -m venv venv_$version
    source venv_$version/bin/activate
    pip install qiskit==$version
    pip install pytest
    pip install memory_profiler
    python3 runner.py $version > memory_$version.txt
    deactivate
done

tail -n 1 memory_${versions[-1]}.txt
