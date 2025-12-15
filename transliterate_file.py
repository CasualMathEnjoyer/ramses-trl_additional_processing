#!/usr/bin/env python3

"""
This script is a copy from /home/katka/hieroglyphs/ramses-trl/python/batch_translit.py
Original repository: https://gitlab2.cnam.fr/rosmorse/ramses-trl

Uses the following functions/classes from the ramses-trl submodule:
- Vocabulary from translit_lib.vocabulary
- Encoder, Decoder, BidiEncoder, Transliterator from translit_lib.encoder_decoder

Efficient batch transliteration using a single network.h5 model file.
Takes an input file, transliterates each line, and outputs the results to a file.

Usage:
    python transliterate_file.py \
        --net network.h5 \
        --input data/src-sep-test.txt \
        --output translit_output.txt
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ramses-trl', 'python'))
from translit_lib.vocabulary import Vocabulary
from translit_lib.encoder_decoder import Encoder, Decoder, BidiEncoder, Transliterator


def build_transliterator(net, hidden=500, arch="bidiseg"):
    """
    Initialise vocabulary and encoder/decoder ONCE.
    Loads model from network.h5.
    """

    vocabulary = Vocabulary()
    vocabulary.loadDefault()

    attention_output = False

    if arch in ("bidi", "bidiseg"):
        encoder = BidiEncoder(input_size=vocabulary.size(), hidden=hidden)
        encoder_model = encoder.load_model_from_file(net)

        decoder = Decoder(hidden=hidden)
        decoder_model = decoder.load_model_from_file(net, attention_output=attention_output)

    elif arch in ("seg", "plain"):
        encoder = Encoder(input_size=vocabulary.size(), hidden=hidden)
        encoder_model = encoder.load_model_from_file(net)

        decoder = Decoder(hidden=hidden)
        decoder_model = decoder.load_model_from_file(net)

    else:
        raise ValueError(f"Unknown architecture '{arch}'")

    transliterator = Transliterator(
        vocabulary,
        encoder_model,
        decoder_model,
        attention_output=attention_output
    )

    return transliterator


def create_unicode_mapping(target_file):
    """
    Create a mapping from character codes to Unicode characters based on training data.
    Since the model was trained with max_code=255, Unicode chars > 255 were mapped to code 3.
    We need to map all Unicode chars > 255 that appear in training data.
    Returns a set of valid Unicode characters that should replace code 3.
    """
    unicode_chars = set()
    with open(target_file, "r", encoding="utf-8") as f:
        for line in f:
            for char in line:
                if ord(char) > 255:
                    unicode_chars.add(char)
    
    return unicode_chars


def batch_transliterate(net, input_path, output_path,
                        arch="bidiseg", hidden=500, beam=1, compact=False, target_file=None):

    transliterator = build_transliterator(net, hidden=hidden, arch=arch)
    
    unicode_chars = set()
    if target_file and os.path.exists(target_file):
        unicode_chars = create_unicode_mapping(target_file)
        print(f"Found {len(unicode_chars)} Unicode characters > 255 in target file")
        print("Note: Model was trained with max_code=255, so Unicode chars were mapped to code 3.")
        print("This means the model cannot distinguish between different Unicode characters.")
        print("The output will contain code 3 (^C) where Unicode chars should be.")
        print("Consider retraining with a higher max_code or proper character mapping.")

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                outfile.write("\n")
                continue

            tr = transliterator.transliterate(line, beam_width=beam)
            
            if chr(3) in tr:
                print(f"Warning: Output contains code 3 (^C) control character. This indicates Unicode characters > 255 were mapped to unknown during training.")
                print(f"Original output: {repr(tr[:100])}")

            if not compact:
                tr = " ".join(tr.replace(" ", "_"))

            outfile.write(tr + "\n")
            print(tr)


def main():
    parser = argparse.ArgumentParser(description="Batch transliteration without Docker.")
    parser.add_argument("--net", required=True, help="Path to network model (.h5)")
    parser.add_argument("--input", required=True, help="Input file of MDC lines")
    parser.add_argument("--output", required=True, help="Output file for transliterations")
    parser.add_argument("--arch", default="bidiseg", help="Network architecture")
    parser.add_argument("--hidden", type=int, default=500, help="Hidden layer size")
    parser.add_argument("--beam", type=int, default=1, help="Beam width")
    parser.add_argument("--compact", action="store_true", help="Do not space-separate characters")
    parser.add_argument("--target-file", help="Target training file to create Unicode mapping (for fixing code 3 -> Unicode chars)")

    args = parser.parse_args()

    batch_transliterate(
        net=args.net,
        input_path=args.input,
        output_path=args.output,
        arch=args.arch,
        hidden=args.hidden,
        beam=args.beam,
        compact=args.compact,
        target_file=args.target_file,
    )


if __name__ == "__main__":
    main()




