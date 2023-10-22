#!/usr/bin/env zsh

# Run the Python script and pipe its output to a text file
python3 runner.py > memory.txt

# Echo the last line of the text file to the terminal
tail -n 1 memory.txt
