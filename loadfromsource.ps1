#!/bin/bash
& c:/Users/tux/Documents/HiCT/hict.venv/Scripts/Activate.ps1
Set-Variable "HICT_DIR" "${PSScriptRoot}/../hict/"
$env:PYTHONPATH += ";${HICT_DIR}"
echo "Setting HICT_DIR = ${HICT_DIR} and PYTHONPATH = ${env:PYTHONPATH}"
