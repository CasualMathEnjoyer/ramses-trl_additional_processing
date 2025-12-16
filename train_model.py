#!/usr/bin/env python3

"""
This script is a modified copy from ramses-trl/python/train_bidi_segment_h500.py
Original repository: https://gitlab2.cnam.fr/rosmorse/ramses-trl

Uses the following functions/classes from the ramses-trl submodule:
- Vocabulary from translit_lib.vocabulary
- CorpusSegmentation, create_and_train from translit_lib.corpus_utils
- EncoderDecoderBidiSegmentation from translit_lib.encoder_decoder

Trains a network with a bidirectional encoder, segmentation output, and hidden layers of width 500.
Takes source and target files as command-line arguments.
"""

import sys
import os
import argparse
import glob
import re

import keras
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth error: {e}", file=sys.stderr)

if hasattr(keras.layers, 'Input') and not hasattr(keras.models, 'Input'):
    keras.models.Input = keras.layers.Input

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ramses-trl', 'python'))
from translit_lib.vocabulary import Vocabulary
from translit_lib.corpus_utils import CorpusSegmentation, _load_character_corpus
import translit_lib.corpus_utils as corpus_utils
from translit_lib.encoder_decoder import EncoderDecoderBidiSegmentation
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping


class CharacterVocabulary:
    """
    A vocabulary for target characters that only includes characters actually used in the data.
    Similar to Vocabulary but for characters instead of words.
    """
    def __init__(self):
        self.MASK = 0
        self.BOS = 1
        self.EOS = 2
        self.UNKNOWN = 3
        
        self._char2idx = {}
        self._idx2char = {}
        self._record_reserved(self.MASK, '<MASK>')
        self._record_reserved(self.BOS, '<BOS>')
        self._record_reserved(self.EOS, '<EOS>')
        self._record_reserved(self.UNKNOWN, '<UNKNOWN>')
        self.last_index = 3
    
    def _record_reserved(self, index, char):
        self._char2idx[char] = index
        self._idx2char[index] = char
    
    def add_char(self, char):
        if char not in self._char2idx:
            self.last_index += 1
            self._char2idx[char] = self.last_index
            self._idx2char[self.last_index] = char
        return self._char2idx[char]
    
    def get_index(self, char):
        return self._char2idx.get(char, self.UNKNOWN)
    
    def get_char(self, index):
        return self._idx2char.get(index, '<UNKNOWN>')
    
    def size(self):
        return self.last_index + 1
    
    def build_from_file(self, file_name, space_separated=True):
        """
        Build vocabulary from target file, collecting all unique characters.
        """
        chars = set()
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n\r\t ')
                if space_separated:
                    for char in line.split(' '):
                        if char:
                            chars.add(char)
                else:
                    for char in line:
                        chars.add(char)
        
        for char in sorted(chars):
            self.add_char(char)
        
        return len(chars)


def build_character_vocabulary(target_file):
    """
    Build a character vocabulary from the target file.
    Returns the vocabulary and its size.
    """
    vocab = CharacterVocabulary()
    num_chars = vocab.build_from_file(target_file, space_separated=True)
    vocab_size = vocab.size()
    print(f"Built character vocabulary: {num_chars} unique characters, vocabulary size: {vocab_size}")
    return vocab, vocab_size


def patch_corpus_utils_for_character_vocab(char_vocab, vocab_size):
    """
    Monkey-patch the corpus_utils module to use a custom character vocabulary instead of Unicode code points.
    """
    import translit_lib.corpus_utils as corpus_utils
    
    original_load_character_corpus = corpus_utils._load_character_corpus
    
    def patched_load_character_corpus(file_name, max_code_param=None, space_separated=True):
        """
        Load character corpus using custom character vocabulary instead of Unicode codes.
        """
        res = []
        with open(file_name, encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n\r\t ")
                if space_separated:
                    chars = [c for c in line.split(" ") if c]
                else:
                    chars = list(line)
                codes = [char_vocab.BOS] + [char_vocab.get_index(c) for c in chars] + [char_vocab.EOS]
                res.append(codes)
        return res
    
    corpus_utils._load_character_corpus = patched_load_character_corpus
    
    original_encode_segmented_corpus = corpus_utils._encode_segmented_corpus
    
    def patched_encode_segmented_corpus(encoded_inputs, encoded_outputs, segmentation, num_input_classes, num_output_classes):
        if num_output_classes == 256:
            num_output_classes = vocab_size
        return original_encode_segmented_corpus(encoded_inputs, encoded_outputs, segmentation, num_input_classes, num_output_classes)
    
    corpus_utils._encode_segmented_corpus = patched_encode_segmented_corpus
    
    original_corpus_segmentation_no_batch = CorpusSegmentation.no_batch
    
    def patched_no_batch(self):
        list_in = self._build_list_in()
        segment_out = self._load_segment_out()
        list_out = self._load_target_corpus()
        return corpus_utils._encode_segmented_corpus(
            list_in, list_out, segment_out,
            self.vocabulary().size(), vocab_size
        )
    
    CorpusSegmentation.no_batch = patched_no_batch
    
    original_corpus_segmentation_generator = CorpusSegmentation.generator
    
    def patched_generator(self, epochs):
        import operator
        import random
        def _mygenerator():
            def extract(a_list, indexes):
                return operator.itemgetter(*indexes)(a_list)
            list_in = self._build_list_in()
            list_out = self._load_target_corpus()
            seg_out = self._load_segment_out()
            epoch_count = 0
            while True:
                indexes = list(range(0, self.corpus_size()))
                random.shuffle(indexes)
                for i in range(0, self.corpus_size(), self.batch_size()):
                    batch_indexes = indexes[i:i + self.batch_size()]
                    batch_in = extract(list_in, batch_indexes)
                    batch_out = extract(list_out, batch_indexes)
                    batch_out_seg = extract(seg_out, batch_indexes)
                    res = corpus_utils._encode_segmented_corpus(
                        batch_in, batch_out, batch_out_seg,
                        self.vocabulary().size(), 
                        vocab_size
                    )
                    yield res
                if epochs is not None and epoch_count >= epochs:
                    break
                else:
                    epoch_count = epoch_count + 1
        return _mygenerator()
    
    CorpusSegmentation.generator = patched_generator


def train_model(src_train, tgt_train, src_val, tgt_val, output_model, batch_size=None):
    """
    Train a bidirectional segmentation model.
    
    Args:
        src_train: Path to source training file (with separations, e.g., src-sep-train.txt)
        tgt_train: Path to target training file (e.g., tgt-train.txt)
        src_val: Path to source validation file (with separations, e.g., src-sep-val.txt)
        tgt_val: Path to target validation file (e.g., tgt-val.txt)
        output_model: Path pattern for output model file (can include {epoch}, {val_softmax_acc}, {val_softmax_loss})
        batch_size: Batch size for training (default: 4, reduced for Unicode support with large vocabulary)
    """
    hidden = 500
    if batch_size is None:
        batch_size = 8
    num_epochs = 200
    period = 20
    
    vocabulary = Vocabulary()
    vocabulary.loadDefault()

    if not os.path.exists(src_train):
        print(f"Error: Source training file not found: {src_train}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(tgt_train):
        print(f"Error: Target training file not found: {tgt_train}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(src_val):
        print(f"Error: Source validation file not found: {src_val}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(tgt_val):
        print(f"Error: Target validation file not found: {tgt_val}", file=sys.stderr)
        sys.exit(1)

    print("Building character vocabulary from target file...")
    char_vocab, vocab_size = build_character_vocabulary(tgt_train)
    
    print("Patching corpus_utils for character vocabulary support...")
    patch_corpus_utils_for_character_vocab(char_vocab, vocab_size)
    
    decoder_output_size = vocab_size
    print(f"Decoder output size: {decoder_output_size} (custom character vocabulary)")

    corpus = CorpusSegmentation(vocabulary=vocabulary, src=src_train, tgt=tgt_train, batch_size=batch_size)
    corpus_val = CorpusSegmentation(vocabulary=vocabulary, src=src_val, tgt=tgt_val, batch_size=batch_size)

    def model_generator():
        encoder_decoder_seg = EncoderDecoderBidiSegmentation(
            hidden=hidden, 
            encoder_input_size=vocabulary.size(), 
            decoder_input_size=decoder_output_size
        )
        model = encoder_decoder_seg.build_model()
        model.summary()
        return model

    output_dir = os.path.dirname(output_model) or '.'
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint-epoch{epoch:03d}.h5')
    best_model_path = output_model.replace('{epoch:03d}', 'best').replace('{val_softmax_acc:.4f}', '').replace('-loss{val_softmax_loss:.4f}', '')
    if best_model_path == output_model:
        best_model_path = output_model.replace('.h5', '_best.h5')
    
    initial_epoch = 0
    if os.path.exists(best_model_path):
        try:
            model = keras.models.load_model(best_model_path)
            print(f"Loaded existing model from: {best_model_path}")
            
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint-epoch*.h5'))
            if checkpoint_files:
                epochs = []
                for f in checkpoint_files:
                    match = re.search(r'epoch(\d+)', f)
                    if match:
                        epochs.append(int(match.group(1)))
                if epochs:
                    initial_epoch = max(epochs)
                    print(f"Found checkpoints up to epoch {initial_epoch}, resuming from epoch {initial_epoch}")
        except Exception as e:
            print(f"Could not load existing model ({e}), starting fresh...")
            model = model_generator()
    else:
        model = model_generator()
    
    class PeriodicCheckpoint(Callback):
        def __init__(self, filepath, period):
            super().__init__()
            self.filepath = filepath
            self.period = period
        
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.period == 0:
                filepath = self.filepath.format(epoch=epoch + 1)
                self.model.save(filepath, overwrite=True)
                print(f"Checkpoint saved: {filepath}")
    
    callbacks = [
        PeriodicCheckpoint(checkpoint_path, period),
        ModelCheckpoint(
            filepath=best_model_path,
            verbose=1,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
    ]
    
    print(f"Starting training...")
    print(f"  Training files: {src_train} -> {tgt_train}")
    print(f"  Validation files: {src_val} -> {tgt_val}")
    print(f"  Best model will be saved to: {best_model_path}")
    print(f"  Checkpoints will be saved to: {checkpoint_dir}/")
    print(f"  Hidden size: {hidden}, Batch size: {batch_size}, Epochs: {num_epochs}")
    print(f"  Checkpoint frequency: every {period} epochs")
    if initial_epoch > 0:
        print(f"  Resuming from epoch {initial_epoch}")
    
    corpus_generator = corpus.generator(num_epochs)
    
    val_batch_size = min(corpus_val.batch_size(), 8)
    
    def validation_generator():
        """Generator for validation data that processes in batches to avoid OOM"""
        import operator
        list_in = corpus_val._build_list_in()
        list_out = corpus_val._load_target_corpus()
        seg_out = corpus_val._load_segment_out()
        
        def extract(a_list, indexes):
            return operator.itemgetter(*indexes)(a_list)
        
        val_size = corpus_val.corpus_size()
        
        for i in range(0, val_size, val_batch_size):
            batch_indexes = list(range(i, min(i + val_batch_size, val_size)))
            batch_in = extract(list_in, batch_indexes)
            batch_out = extract(list_out, batch_indexes)
            batch_out_seg = extract(seg_out, batch_indexes)
            res = corpus_utils._encode_segmented_corpus(
                batch_in, batch_out, batch_out_seg,
                corpus_val.vocabulary().size(), vocab_size
            )
            yield res
    
    val_steps = (corpus_val.corpus_size() + val_batch_size - 1) // val_batch_size
    
    try:
        model.fit_generator(
            generator=corpus_generator,
            steps_per_epoch=corpus.steps_per_epoch(),
            validation_data=validation_generator(),
            validation_steps=val_steps,
            epochs=num_epochs,
            callbacks=callbacks,
            initial_epoch=initial_epoch,
            max_queue_size=1,
            workers=1,
            use_multiprocessing=False
        )
        print(f"Training completed. Best model saved to: {best_model_path}")
    except KeyboardInterrupt:
        print(f"\nTraining interrupted. Latest checkpoint saved in: {checkpoint_dir}/")
        print(f"To resume, use the same command - it will load from: {best_model_path}")
        raise
    except Exception as e:
        print(f"\nTraining error: {e}")
        print(f"Latest checkpoint saved in: {checkpoint_dir}/")
        raise
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a bidirectional segmentation model for transliteration."
    )
    parser.add_argument("--src-train", required=True, 
                       help="Path to source training file (with separations, e.g., src-sep-train.txt)")
    parser.add_argument("--tgt-train", required=True, 
                       help="Path to target training file (e.g., tgt-train.txt)")
    parser.add_argument("--src-val", required=True, 
                       help="Path to source validation file (with separations, e.g., src-sep-val.txt)")
    parser.add_argument("--tgt-val", required=True, 
                       help="Path to target validation file (e.g., tgt-val.txt)")
    parser.add_argument("--output", required=True, 
                       help="Path pattern for output model file (can include {epoch}, {val_softmax_acc}, {val_softmax_loss})")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for training (default: 8, reduced for Unicode support with large vocabulary)")

    args = parser.parse_args()

    train_model(
        src_train=args.src_train,
        tgt_train=args.tgt_train,
        src_val=args.src_val,
        tgt_val=args.tgt_val,
        output_model=args.output,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

