"""
This script is a modified copy from ramses-trl/python/average_levenshtein_distance.py
Original repository: https://gitlab2.cnam.fr/rosmorse/ramses-trl

Modifications:
- Changed from corpus-based target to direct file path via --target argument
- Made it work with any file pair, not just specific corpus files
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ramses-trl', 'python'))
from translit_lib.levenshtein import LevenshteinDistance


def compare(computed_path, target_path):
    """
    Compare two files using Levenshtein distance.
    
    Args:
        computed_path: Path to the computed/actual output file
        target_path: Path to the target/gold standard file
    """
    if not os.path.exists(computed_path):
        print(f"Error: Computed file not found: {computed_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(target_path):
        print(f"Error: Target file not found: {target_path}", file=sys.stderr)
        sys.exit(1)
    
    distance = LevenshteinDistance()
    distance.fullDist(target_path, computed_path)
    res = distance.meanDist(target_path, computed_path)
    print('''please note that the distance can be larger than l, 
    because l is only the length of the expected output, 
    whereas the actual output might be larger (especially if it starts repeating itself)
    Legend : 
      - l : size of the gold output
      - n : number of sentences of size l
      - s : sum of levenshtein distances between gold and actual output
      - m : s/(n*l) estimation of levenshtein distance by sign for length l
      - N : total number of *signs* seen in output
      - S : total error from now
      - M : S / N
    ''')
    print("l\tn\ts\tm\tN\tS\tM")
    for key, value in sorted(res.items()):        
        print("{}\t{}\t{:7.3f}\t{:7.3f}\t{}\t{:7.3f}\t{:7.3f}".format(key, *value))


def _startSoft():
    usage = '''
    Computes Average Levenshtein Distance between a computed result and a target file.
    Works with any pair of text files, comparing them line by line.
    '''
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument("path", help='path of the computed transliteration file to evaluate')
    parser.add_argument("--target", dest="target", required=True, 
                       help="path to the target/gold standard file to compare against")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()
    compare(args.path, args.target)


if __name__ == "__main__":
    _startSoft()

