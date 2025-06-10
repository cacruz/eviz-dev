#!/bin/bash

function prompt() {

read -p "Press Y to continue: " -n 1 -r

if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo
  return
else
  exit 1
fi
}

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
#    files=$(find "$EVIZ_CONFIG_PATH" -name \*.yaml -print)

#    for f in "${files[@]}"; do
#	sed -i -r 's@EVIZ_DATA_PATH@'"$EVIZ_DATA_PATH"'@' "$f"
#	sed -i -r 's@EVIZ_OUTPUT_PATH@'"$EVIZ_OUTPUT_PATH"'@' "$f"
#    done

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
    echo "Usage ./integ.sh [conda env name]"
    echo "Make sure to activate the viz environment before running this script."
    exit 1
fi

echo "Single-plot tests"
echo "------------------"
echo
# Use -f option (would override EVIZ_CONFIG_PATH)
f_option=/Users/ccruz/projects/Eviz/config/simple/simple.yaml
#echo "Gridded no specs"
#python autoviz.py -s gridded -f $f_option -v 0
#echo

# Use -c option (would override EVIZ_CONFIG_PATH)
c_option=/Users/ccruz/projects/Eviz/config
echo "Using config -c $c_option"
echo "Source: 'gridded' default specs"
python autoviz.py -s gridded -c $c_option -v 0
echo

echo "Source: 'gridded' different (multiple) sources"
python autoviz.py -s gridded -f $EVIZ_CONFIG_PATH/gridded/gridded_multiple.yaml -v 0
echo

echo "Source: 'gridded' OpenDAP access"
python autoviz.py -s gridded -f $EVIZ_CONFIG_PATH/gridded/gridded_opendap.yaml -v 0
echo

echo "Source: 'geos', with 4 files"
python autoviz.py -s geos -v 0
echo

echo "Source: 'ccm', with 2 files"
python autoviz.py -s ccm -v 0
echo

echo "Source: 'ccm', 2 files and unit conversions"
python autoviz.py -s ccm -v 0 -f $EVIZ_CONFIG_PATH/ccm/ccm_multiple.yaml
echo

echo "Source: 'wrf', with 1 file (multiple 2D fields)"
python autoviz.py -s wrf -v 0
echo

echo "Source: 'wrf', GIF option"
python autoviz.py -s wrf -v 0 -f $EVIZ_CONFIG_PATH/wrf_gif.yaml
echo


echo "Source: 'lis', with 1 file (multiple 2D fields)"
python autoviz.py -s lis -v 0
echo

echo "Source: 'airnow', with 12 files"
python autoviz.py -s airnow -v 0
echo

echo "Source: 'omi', with 1 file"
python autoviz.py -s omi -v 0
echo

#echo "Source: mopitt, with 1 file"
#python autoviz.py -s mopitt -v 0

#echo "Source: landsat, with 1 file"
#python autoviz.py -s landsat -v 0

#echo "Source: omi,airnow"
#python autoviz.py -s omi,airnow -v 0

#echo "Source: wrf,lis"
#python autoviz.py -s wrf,lis -v 0

#echo "Source: gridded,geos,cf"
#python autoviz.py -s gridded,geos,cf -v 0

echo "Comparison-plot tests"
echo "---------------------"
echo
export EVIZ_CONFIG_PATH=/Users/ccruz/scratch/eviz/compare/config
echo "Using config path:"
echo "$EVIZ_CONFIG_PATH"
echo

echo "Source: 'ccm' vs 'ccm' (compare)"
python autoviz.py -s ccm -v 0 -f  $EVIZ_CONFIG_PATH/ccm/ccm_compare.yaml
echo

echo "Source: 'ccm' vs 'ccm' (compare-diff 3x1)"
python autoviz.py -s ccm -v 0 -f  $EVIZ_CONFIG_PATH/ccm/ccm_compare_3x1.yaml
echo

echo "Source: 'ccm' vs 'ccm' (compare-diff 2x2)"
python autoviz.py -s ccm -v 0 -f  $EVIZ_CONFIG_PATH/ccm/ccm_compare_2x2.yaml
echo

echo "Source: 'ccm' vs 'ccm' (overlay)"
python autoviz.py -s ccm -v 0 -f  $EVIZ_CONFIG_PATH/ccm/ccm_compare_overlay.yaml
echo

echo "Source: 'wrf', with 2 files (compare)"
python autoviz.py -s wrf -v 0 -f $EVIZ_CONFIG_PATH/wrf_compare.yaml
echo

echo "Source: 'wrf', line-plot (compare)"
python autoviz.py -s wrf -v 0 -f $EVIZ_CONFIG_PATH/wrf_xt_compare.yaml
echo

echo "Source: lis, with 2 files (compare multiple fields)"
python autoviz.py -s lis -v 0
echo

#echo "Source: 'ccm' vs 'omi' "
#python autoviz.py -s ccm,omi -v 0

#echo "Source: ccm,mopitt"
#python autoviz.py -s ccm,mopitt -v 0

#echo "Source: cf,geos (geos=merra2)"
#python autoviz.py -s cf,geos -v 0

#echo "Source: cf,airnow"
#python autoviz.py -s cf,airnow -v 0

#echo "Source: cf,omi"
#python autoviz.py -s cf,omi -v 0

#echo "Source: cf,geos (comparison cf,merra2 + single cf)"
#python autoviz.py -s cf,geos -v 0


# Altair backend

c_option=/Users/ccruz/projects/Eviz/config
echo "Source: 'gridded' - altair backend"
python autoviz.py -s gridded -f $EVIZ_CONFIG_PATH/gridded/gridded_altair.yaml -v 0
echo

# Hvplot backend

c_option=/Users/ccruz/projects/Eviz/config
echo "Source: 'gridded' - hvplot backend"
python autoviz.py -s gridded -f $EVIZ_CONFIG_PATH/gridded/gridded_hvplot.yaml -v 0
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
