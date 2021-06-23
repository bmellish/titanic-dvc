import io
import os
import random
import re
import sys
import xml.etree.ElementTree

import yaml

params = yaml.safe_load(open("params.yaml"))["prepare"]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)

# Test data set split ratio
split = params["split"]
random.seed(params["seed"])

input_train = sys.argv[1]
input_test = sys.argv[2]
output_train = os.path.join("processed_data", "prepared", "train.tsv")
output_test = os.path.join("processed_data", "prepared", "test.tsv")


def process_posts(fd_in_test, fd_in_train, fd_out_train, fd_out_test):
    num = 1
    for line in fd_in_test:
        try:
            fd_out_test.write(line)
            num += 1
        except Exception as ex:
            sys.stderr.write("Skipping the broken line {num}: {ex}".format(num, ex))
    num = 1
    for line in fd_in_train:
        try:
            fd_out_train.write(line)
            num += 1
        except Exception as ex:
            sys.stderr.write("Skipping the broken line {num}: {ex}".format(num, ex))


os.makedirs(os.path.join("processed_data", "prepared"), exist_ok=True)

try:
    with io.open(input_train, encoding="utf8") as fd_in_test:
        with io.open(input_train, encoding="utf8") as fd_in_train:
            with io.open(output_train, "w", encoding="utf8") as fd_out_train:
                with io.open(output_test, "w", encoding="utf8") as fd_out_test:
                    process_posts(fd_in_test, fd_in_train, fd_out_train, fd_out_test)
except IOError as ex:
    print(ex)
