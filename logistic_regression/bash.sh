#! /bin/bash
#
# file: next.sh
#
# This bash script analyzes the record named in its command-line
# argument ($1), and writes per-sample classification to the file
# "$1.vec".  This script is run once for each record in the Challenge
# test set.  The input record ($1.hea and $1.mat) will be located in
# the current working directory.
#
# The output file must be a plain text file, with one line for each
# sample of the record.  Each line must be a number between 0 and 1,
# indicating the probability that an arousal is occurring at that
# instant.  For an (unrealistic) example, if an arousal appears to
# begin 15 milliseconds after the start of the record, the $RECORD.vec
# file might contain:
#
# 0.000
# 0.002
# 0.048
# 0.955
# 1.000
# 0.946

set -e
set -o pipefail

RECORD=$1

./run_my_classifier.py "$RECORD"
