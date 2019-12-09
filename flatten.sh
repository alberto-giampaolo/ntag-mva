#!/usr/bin/env bash

for i in {7..19..3}; do
    python3 flatten_ttree.py $i $((i+3))
done
