#!/bin/bash
conda activate tf2
python danielsawyer_project-1.py |& tee output.txt
conda deactivate
