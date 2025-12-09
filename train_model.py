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
from translit_lib.corpus_utils import CorpusSegmentation
from translit_lib.encoder_decoder import EncoderDecoderBidiSegmentation
from keras.callbacks import ModelCheckpoint, Callback


def train_model(src_train, tgt_train, src_val, tgt_val, output_model):
    """
    Train a bidirectional segmentation model.
    
    Args:
        src_train: Path to source training file (with separations, e.g., src-sep-train.txt)
        tgt_train: Path to target training file (e.g., tgt-train.txt)
        src_val: Path to source validation file (with separations, e.g., src-sep-val.txt)
        tgt_val: Path to target validation file (e.g., tgt-val.txt)
        output_model: Path pattern for output model file (can include {epoch}, {val_softmax_acc}, {val_softmax_loss})
    """
    hidden = 500
    batch_size = 16
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

    corpus = CorpusSegmentation(vocabulary=vocabulary, src=src_train, tgt=tgt_train, batch_size=batch_size)
    corpus_val = CorpusSegmentation(vocabulary=vocabulary, src=src_val, tgt=tgt_val, batch_size=batch_size)

    def model_generator():
        encoder_decoder_seg = EncoderDecoderBidiSegmentation(
            hidden=hidden, 
            encoder_input_size=vocabulary.size(), 
            decoder_input_size=256
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
            initial_epoch = getattr(model, 'epoch', 0) if hasattr(model, 'epoch') else 0
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
    
    try:
        model.fit_generator(
            generator=corpus_generator,
            steps_per_epoch=corpus.steps_per_epoch(),
            validation_data=corpus_val.no_batch(),
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

    args = parser.parse_args()

    train_model(
        src_train=args.src_train,
        tgt_train=args.tgt_train,
        src_val=args.src_val,
        tgt_val=args.tgt_val,
        output_model=args.output
    )


if __name__ == "__main__":
    main()

