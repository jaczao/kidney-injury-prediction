#!/usr/bin/env python3

import argparse
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/test.csv")
    parser.add_argument("--output", default="data/aki.csv")
    flags = parser.parse_args()
    r = csv.reader(open(flags.input))
    w = csv.writer(open(flags.output, "w"))
    w.writerow(("aki",))
    next(r) # skip headers
    for _ in r:
        # TODO: Implement a better model
        w.writerow(("y",))

if __name__ == "__main__":
    main()