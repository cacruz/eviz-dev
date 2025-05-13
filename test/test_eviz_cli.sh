#!/bin/bash
# A script to test the eviz.py command line interface options
# Assume EVIZ_CONFIG_PATH is defined in the environment

[[ ! -e $EVIZ_CONFIG_PATH ]] && exit
echo "Basic call (no config, no configfile)"
echo "Should use config files specified in $EVIZ_CONFIG_PATH"
python eviz.py -s lis -v 0

# Use -c option (overrides EVIZ_CONFIG_PATH)
c_option=/Users/ccruz/scratch/eviz/config
echo "Specify config -c $c_option"
python eviz.py -s lis -c $c_option -v 0

# Use -f option (overrides EVIZ_CONFIG_PATH)
f_option=/Users/ccruz/scratch/eviz/config/lis/lis.yaml
echo "Specify configfile -f $f_option"
python eviz.py -s lis -f $f_option -v 0

unset EVIZ_CONFIG_PATH
echo "No config files at all (should break on MAC, OK on DISCOVER)"
echo "Should use config files specified in config/"
python eviz.py -s lis -v 0
