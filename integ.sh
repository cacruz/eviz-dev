#!/bin/bash
set -e

if [[ -z "$EVIZ_CONFIG_PATH" ]]; then
    echo "Environment variable EVIZ_CONFIG_PATH must be defined"
    exit 1
fi
echo "Found EVIZ_CONFIG_PATH: $EVIZ_CONFIG_PATH"
if [[ ! -d "$EVIZ_CONFIG_PATH" ]]; then
    echo "Error: $EVIZ_CONFIG_PATH does not exist"
    exit 1
fi

os_type=$(uname)
machine_name=$(hostname)

#---------------------------------------------------
# Use templates (in progress)
skip=1

if [ $skip -eq 1 ]; then

    if [[ "$machine_name" == *"discover"* || "$machine_name" == *"borg"* ]]; then
	export EVIZ_DATA_PATH=/discover/nobackup/ccruz/data/eviz/sample_data
	export EVIZ_OUTPUT_PATH=/discover/nobackup/$USER/scratch/output
    else   # Mac
	export EVIZ_DATA_PATH=/Users/$USER/data/eviz
	export EVIZ_OUTPUT_PATH=/Users/$USER/scratch/eviz/output_plots
	if [ -z "$EVIZ_DATA_PATH" ]; then
	    echo "Please set EVIZ_DATA_PATH"
	    exit 1
	fi
	if [ -z "$EVIZ_OUTPUT_PATH" ]; then
	    echo "Please set EVIZ_OUTPUT_PATH"
	    exit 1
	fi
    fi
    files=$(find $EVIZ_CONFIG_PATH -name \*.yaml -print)

    for f in ${files[@]}; do
	sed -i -r 's@EVIZ_DATA_PATH@'"$EVIZ_DATA_PATH"'@' $f
	sed -i -r 's@EVIZ_OUTPUT_PATH@'"$EVIZ_OUTPUT_PATH"'@' $f
    done

fi
#---------------------------------------------------

if [ "$os_type" == "Darwin" ]; then
    echo "Running tests on Darwin OS"
elif [ "$os_type" == "Linux" ]; then
    echo "Running tests on Linux OS"
else
    echo "Unsupported operating system: $os_type"
    exit 1
fi

eviz_env=$1
current_env=$CONDA_DEFAULT_ENV
if [ "$current_env" == "$eviz_env" ]; then
    echo "Using conda env $eviz_env."
else
    echo "Error: Activate the $eviz_env environment before running this script."
    exit 1
fi

echo "Basic 'gridded' tests"
echo "---------------------"
echo
# Use -f option (would override EVIZ_CONFIG_PATH)
f_option=/Users/ccruz/projects/Eviz/config/simple/simple.yaml
echo "Using configfile -f $f_option"
python autoviz.py -s gridded -f $f_option -v 0
echo

# Use -f option (would override EVIZ_CONFIG_PATH)
f_option=/Users/ccruz/projects/Eviz/config/simple_spec/sample.yaml
echo "Using configfile -f $f_option"
echo "and its associated spec file"
python autoviz.py -s gridded -f $f_option -v 0
echo

# Use -c option (would override EVIZ_CONFIG_PATH)
c_option=/Users/ccruz/projects/Eviz/config
echo "Using config -c $c_option"
echo "(expects app and spec under gridded)"
python autoviz.py -s gridded -c $c_option -v 0
echo


echo "Same as above, but using EVIZ_CONFIG_PATH"
python autoviz.py -s gridded -v 0
echo

echo "Single-plot tests"
echo "-----------------"
echo

echo "Source: geos, with 4 files"
python autoviz.py -s geos -v 0
echo

echo "Source: ccm, with 2 files"
python autoviz.py -s ccm -v 0
echo

echo "Source: ccm, 2 files and unit conversions"
python autoviz.py -s ccm -v 0 -f /Users/ccruz/projects/Eviz/config/ccm/ccm_multiple.yaml
echo

echo "Source: cf, with 3 files"
python autoviz.py -s cf -v 0
echo

echo "Source: wrf, with 1 file"
python autoviz.py -s wrf -v 0
echo

echo "Source: lis, with 1 file"
python autoviz.py -s lis -v 0
echo

echo "Source: airnow, with 12 files"
python autoviz.py -s airnow -v 0
echo

echo "Source: omi, with 1 file"
python autoviz.py -s omi -v 0
echo

echo "Source: mopitt, with 1 file"
#python autoviz.py -s mopitt -v 0
echo

echo "Source: landsat, with 1 file"
#python autoviz.py -s landsat -v 0
echo

echo "Source: omi,airnow"
#python autoviz.py -s omi,airnow -v 0
echo

echo "Source: wrf,lis"
#python autoviz.py -s wrf,lis -v 0
echo

echo "Source: gridded,geos,cf"
#python autoviz.py -s gridded,geos,cf -v 0
echo

echo "Comparison-plot tests"
echo "---------------------"
echo
export EVIZ_CONFIG_PATH=/Users/ccruz/scratch/eviz/compare/config
echo "Using config path:"
echo "$EVIZ_CONFIG_PATH"
echo

echo "Source: ccm vs ccm"
python autoviz.py -s ccm -v 0 -f /Users/ccruz/projects/Eviz/config/ccm/ccm_compare.yaml
echo

echo "Source: ccm,omi"
#python autoviz.py -s ccm,omi -v 0
echo

echo "Source: ccm,mopitt"
#python autoviz.py -s ccm,mopitt -v 0
echo

echo "Source: cf,geos (geos=merra2)"
#python autoviz.py -s cf,geos -v 0
echo

echo "Source: cf,airnow"
#python autoviz.py -s cf,airnow -v 0
echo

echo "Source: cf,omi"
#python autoviz.py -s cf,omi -v 0
echo

echo "Source: cf,geos (comparison cf,merra2 + single cf)"
#python autoviz.py -s cf,geos -v 0
echo

echo "End of tests"
echo "------------"
echo

machine_name=$(hostname)
if [[ "$machine_name" == *"discover"* || "$machine_name" == *"borg"* ]]; then
    unset EVIZ_CONFIG_PATH
    echo "No config files at all (should break on MAC, OK on DISCOVER)"
    echo "Should use config files specified in config/"
    python autoviz.py -s gridded -v 0
fi
