#!/usr/bin/env bash

shell_dir=$(dirname "$0")
project_dir=$(cd "$shell_dir/.."; pwd)
src_dir=$project_dir/src
bin_dir=$project_dir/bin
data_dir=$project_dir/data
out_dir=$project_dir/out
java_exe=java
javac_exe=javac
python3_exe=/opt/homebrew/Cellar/python@3.10/3.10.2/bin/python3

# Delete old features and output
rm -f $out_dir/%-test.feature $out_dir/%-test.chunk

# Create out directory if it doesn't exist
mkdir -p $out_dir

# Run the feature selector
$python3_exe $src_dir/main.py -s $data_dir/%-test $out_dir/%-test.feature

# Run the labeler
$java_exe -Xmx16g -cp $bin_dir/.:$bin_dir/maxent-3.0.0.jar:$bin_dir/trove.jar MEtag $out_dir/%-test.feature $out_dir/model.chunk $out_dir/%-test.chunk
