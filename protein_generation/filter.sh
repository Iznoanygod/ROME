#!/bin/bash
# Usage: ./filter.sh interpro.dat "super family name" > sf_ids.txt

INTERPRO_FILE=$1
SUPERFAMILY=$2

grep "$SUPERFAMILY" "$INTERPRO_FILE" \
    | awk '{print $1}' \
    | sort -u \
    | awk '{print "AF-" $1 "-F1-model_v6"}'
