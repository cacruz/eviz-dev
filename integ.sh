#!/bin/bash

function prompt() {
    exp="$1"
    echo Running $exp
    if [ "$PROMPT_ENABLED" -eq 1 ]; then
        read -p "Press Y to run, or any other key to skip: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping.."
            return 1
        fi
    fi
    return 0
}


set -e

# Command-line args
PROMPT_ENABLED=1
if [[ "$2" == "--no-prompt" ]]; then
    PROMPT_ENABLED=0
fi

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

# Use these options to override EVIZ_CONFIG_PATH
c_option=/Users/ccruz/projects/Eviz/config
f_option=/Users/ccruz/projects/Eviz/config/simple/simple.yaml

if prompt "Gridded no specs"; then
    python autoviz.py -s gridded -f $f_option -v 0
    echo
fi

if prompt "Source: 'gridded' default specs"; then
    python autoviz.py -s gridded -c $c_option -v 0
    echo
fi

if prompt "Source: 'gridded' multiple sources"; then
    python autoviz.py -s gridded -f $EVIZ_CONFIG_PATH/gridded/gridded_multiple.yaml -v 0
    echo
fi

if  prompt "Source: 'gridded' OpenDAP access"; then
    python autoviz.py -s gridded -f $EVIZ_CONFIG_PATH/gridded/gridded_opendap.yaml -v 0
    echo
fi
    
if prompt "Source: 'geos', MERRA2 averaging"; then
    python autoviz.py -s geos -v 0
    echo
fi
   
if prompt  "Source: 'ccm', with 2 files"; then
   python autoviz.py -s ccm -v 0
   echo
fi
   
if prompt "Source: 'ccm', 2 files and unit conversions"; then
   python autoviz.py -s ccm -v 0 -f $EVIZ_CONFIG_PATH/ccm/ccm_multiple.yaml
   echo
fi
   
if prompt "Source: 'wrf', with 1 file (multiple 2D fields)"; then
   python autoviz.py -s wrf -v 0
   echo
fi
   
if prompt "Source: 'wrf', GIF option"; then
   python autoviz.py -s wrf -v 0 -f $EVIZ_CONFIG_PATH/wrf/wrf_gif.yaml
   echo
fi

if prompt  "Source: 'lis', with 1 file (multiple 2D fields)"; then
   python autoviz.py -s lis -v 0
   echo
fi

if prompt "Source: 'airnow', with 12 files"; then
   python autoviz.py -s airnow -v 0
   echo
fi
   
if prompt "Source: 'omi', with 1 file"; then
   python autoviz.py -s omi -v 0
   echo
fi

if prompt "Source: 'grib'"; then
    python autoviz.py -s grib -v 0
    echo
fi
 
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

if prompt "Source: 'ccm' vs 'ccm' (compare)"; then
   python autoviz.py -s ccm -v 0 -f  $EVIZ_CONFIG_PATH/ccm/ccm_compare.yaml
   echo
fi
   
if prompt "Source: 'ccm' vs 'ccm' (compare-diff 3x1)"; then
   python autoviz.py -s ccm -v 0 -f  $EVIZ_CONFIG_PATH/ccm/ccm_compare_3x1.yaml
   echo
fi
   
if prompt "Source: 'ccm' vs 'ccm' (compare-diff 2x2)"; then
   python autoviz.py -s ccm -v 0 -f  $EVIZ_CONFIG_PATH/ccm/ccm_compare_2x2.yaml
   echo
fi
   
if prompt "Source: 'ccm' vs 'merra2' (compare-diff 3x1)"; then
   python autoviz.py -s ccm -v 0 -f  $EVIZ_CONFIG_PATH/ccm_merra2/app.yaml
   echo
fi
   
if prompt "Source: 'ccm' vs 'ccm' (overlay)"; then
   python autoviz.py -s ccm -v 0 -f  $EVIZ_CONFIG_PATH/ccm/ccm_compare_overlay.yaml
   echo
fi
   
if prompt "Source: 'wrf', with 2 files (compare)"; then
   python autoviz.py -s wrf -v 0 -f $EVIZ_CONFIG_PATH/wrf_compare.yaml
   echo
fi
   
if prompt "Source: 'wrf', line-plot (compare)"; then
   python autoviz.py -s wrf -v 0 -f $EVIZ_CONFIG_PATH/wrf_xt_compare.yaml
   echo
fi
   
if prompt "Source: lis, with 2 files (compare multiple fields)"; then
   python autoviz.py -s lis -v 0 -f $EVIZ_CONFIG_PATH/lis_compare.yaml
   echo
fi
   
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

if prompt "Source: 'gridded' - altair backend"; then
   python autoviz.py -s gridded -f $EVIZ_CONFIG_PATH/gridded/gridded_altair.yaml -v 0
   echo
fi
   
# Hvplot backend

if prompt "Source: 'gridded' - hvplot backend"; then
   python autoviz.py -s gridded -f $EVIZ_CONFIG_PATH/gridded/gridded_hvplot.yaml -v 0
   echo
fi


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
