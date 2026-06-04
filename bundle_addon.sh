#!/bin/sh


abort_bundle() {
  echo "$1"  # input parameter
  echo "aborting bundle"
  cd ..
  kill -INT $$  # the equivalent of ctrl + c
}

cd intmorphs || { echo "cd failed"; return 1; }

# all pycache files have to be deleted before we can zip the .addon file
find . -regex '^.*\(__pycache__\|\.py[co]\)$' -delete

# explicitly importing 'intmorphs' breaks the addon
if grep -rqE --exclude-dir=deps "^(from|import) intmorphs"; then
  abort_bundle "found explicit imports"
fi

# set DEV_MODE = False
found_dev_mode=$(grep -cE "DEV_MODE: bool = (True|False)" intmorphs_globals.py)
if [ "$found_dev_mode" != 1 ]; then
  abort_bundle "the number of DEV_MODE instances is not 1"
fi
sed -i 's/DEV_MODE: bool = True/DEV_MODE: bool = False/g' intmorphs_globals.py

# Find and store the IntMorphs version number. Will be used in .addon file name
version="v$(grep -Po '__version__ = "\K[^"]*' intmorphs_globals.py)"
version=$(echo "$version" | tr . -)  # replace . with - for filename to work

# meta.json is the local user's customized version of config.json, don't bundle this.
rm meta.json

zip -r ../intmorphs-"$version".ankiaddon ./*

cd ..
