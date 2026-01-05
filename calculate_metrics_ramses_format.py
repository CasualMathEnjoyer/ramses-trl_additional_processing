#!/usr/bin/env python3
"""
Calculate transliteration metrics (Levenshtein distance, ROUGE-L, SacreBLEU, GOLD, Character Accuracy)
for RAMSES transliteration output.

Takes prediction and target files as input and outputs metrics in RAMSES format.
"""

import argparse
import string
import os
import sys

# Only use old API (datasets) - crash if not available
import datasets
load_metric = datasets.load_metric

# Import Levenshtein distance from ramses-trl (relative import)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ramses-trl', 'python'))
from translit_lib.levenshtein import LevenshteinDistance


def normalize_tokens(text: str):
    """
    Normalize text for metrics computation.
    Matches GitHub implementation: strip punctuation, lowercase, split into tokens.
    For RAMSES dataset: underscores are word separators, so replace them with spaces first.
    """
    if text is None:
        return []
    # In RAMSES, underscores separate words, so replace with spaces
    text = text.replace("_", " ")
    # Match GitHub: strip(string.punctuation).lower().split()
    text = text.strip(string.punctuation).lower()
    return text.split()


def count_words(lines):
    """
    Count tokens after normalization.
    """
    total = 0
    for line in lines:
        tokens = normalize_tokens(line)
        total += len(tokens)
    return total


def calculate_levenshtein_distance(predictions, targets):
    """
    Calculate average Levenshtein distance per valid token.
    Matches the calculation from the transformer pipeline:
    avg_RLEV_per_valid_char = all_ros_levenstein / all_valid_chars
    Note: "all_valid_chars" in the log actually refers to token count (sum of len(valid) arrays),
    not character count. Tokens are space-separated.
    """
    distance_calc = LevenshteinDistance()
    
    total_distance = 0
    total_valid_tokens = 0
    
    for pred, tgt in zip(predictions, targets):
        # Remove spaces for distance calculation (as done in the original)
        lev_dist = distance_calc.compute(pred, tgt, remove_spaces=True)
        total_distance += lev_dist
        
        # Count tokens (space-separated) in target
        # This matches "all_valid_chars" which is actually the sum of len(valid) token arrays
        tokens = tgt.split()
        total_valid_tokens += len(tokens)
    
    if total_valid_tokens == 0:
        return 0.0, 0.0, 0.0
    
    avg_levenshtein_per_valid = total_distance / total_valid_tokens
    avg_levenshtein_per_line = total_distance / len(predictions) if len(predictions) > 0 else 0.0
    
    # Also calculate per prediction token
    total_pred_tokens = sum(len(pred.split()) for pred in predictions)
    avg_levenshtein_per_pred = total_distance / total_pred_tokens if total_pred_tokens > 0 else 0.0
    
    return avg_levenshtein_per_valid, avg_levenshtein_per_line, avg_levenshtein_per_pred


def calculate_character_accuracy(predictions, targets):
    """
    Calculate character-level accuracy.
    Character accuracy = correct_characters / total_characters_in_target
    
    Uses Levenshtein distance to find the optimal alignment, then counts
    matching characters. This matches the approach used in the transformer pipeline.
    """
    distance_calc = LevenshteinDistance()
    
    total_chars = 0
    correct_chars = 0
    
    for pred, tgt in zip(predictions, targets):
        # Remove spaces for character-level comparison
        pred_chars = pred.replace(" ", "").replace("_", "")
        tgt_chars = tgt.replace(" ", "").replace("_", "")
        
        total_chars += len(tgt_chars)
        
        # Calculate Levenshtein distance to get edit operations
        # Character accuracy = (total_chars - substitutions - insertions - deletions) / total_chars
        # Or more simply: (total_chars - lev_distance) / total_chars
        # But we need to be careful about insertions/deletions
        
        # Simple approach: count matching characters at aligned positions
        # For better accuracy, we could use dynamic programming alignment
        # But for now, use a simpler approach that matches common implementations
        
        # Count exact character matches (position-by-position)
        min_len = min(len(pred_chars), len(tgt_chars))
        matches = sum(1 for i in range(min_len) if pred_chars[i] == tgt_chars[i])
        
        # If strings are same length, all matches count
        # If different lengths, we need to account for that
        if len(pred_chars) == len(tgt_chars):
            correct_chars += matches
        else:
            # For different lengths, use a more sophisticated approach
            # Count matches and penalize for length differences
            # This is a simplified version - actual implementation might use alignment
            lev_dist = distance_calc.compute(pred, tgt, remove_spaces=True)
            # Approximate: correct chars = total - edit distance
            # But this can be negative, so we cap it
            correct_for_line = max(0, len(tgt_chars) - lev_dist)
            correct_chars += correct_for_line
    
    if total_chars == 0:
        return 0.0
    
    return correct_chars / total_chars


def calculate_gold(predictions, targets):
    """
    Calculate GOLD metric: number of sentences translated with no errors at all.
    A sentence is "gold" if prediction exactly matches target (after stripping).
    """
    gold_count = 0
    
    for pred, tgt in zip(predictions, targets):
        if pred.strip() == tgt.strip():
            gold_count += 1
    
    return gold_count


def calculate_rouge_and_bleu(predictions, targets):
    """
    Calculate ROUGE-L and SacreBLEU metrics at token level (word level).
    Matches the implementation from inference_local_model.py:
    - Converts character-separated format to word-level format
    - Removes spaces between characters, replaces underscores with spaces
    - Then normalizes: strip punctuation, lowercase, split into tokens
    """
    if len(predictions) == 0:
        return None, None
    
    # Load metrics
    rouge_metric = load_metric("rouge")
    sacrebleu_metric = load_metric("sacrebleu")
    
    # Process and add batches - convert to word-level format then normalize
    for pred, tgt in zip(predictions, targets):
        # Step 1: Remove spaces between characters to get word-level format
        # "x a a _ = k _ a n x" -> "xaa_=k_anx"
        pred_words = pred.replace(" ", "")
        tgt_words = tgt.replace(" ", "")
        
        # Step 2: Replace underscores with spaces
        # "xaa_=k_anx" -> "xaa =k anx"
        pred_words = pred_words.replace("_", " ")
        tgt_words = tgt_words.replace("_", " ")
        
        # Step 3: Normalize same way as inference_local_model.py
        # Strip punctuation, lowercase, split into tokens
        pred_tokens = pred_words.strip(string.punctuation).lower().split()
        tgt_tokens = tgt_words.strip(string.punctuation).lower().split()
        
        # Pass tokenized lists directly (matching inference_local_model.py)
        # datasets library accepts both lists and strings, but lists give different results
        sacrebleu_metric.add_batch(
            predictions=[pred_tokens],
            references=[[tgt_tokens]]
        )
        
        # ROUGE also accepts tokenized lists
        rouge_metric.add_batch(
            predictions=[pred_tokens],
            references=[[tgt_tokens]]
        )
    
    # Compute metrics
    rouge_result = rouge_metric.compute()
    sacrebleu_result = sacrebleu_metric.compute()
    
    # Extract scores
    sacrebleu = sacrebleu_result["score"]
    
    rougeL_value = rouge_result["rougeL"]
    if isinstance(rougeL_value, (int, float)):
        # evaluate library returns float (0-1)
        rouge_l = 100 * float(rougeL_value)
    else:
        # datasets library returns object with .mid.fmeasure
        rouge_l = 100 * rougeL_value.mid.fmeasure
    
    return rouge_l, sacrebleu


def main():
    parser = argparse.ArgumentParser(
        description="Calculate transliteration metrics (Levenshtein, ROUGE-L, SacreBLEU, GOLD, Character Accuracy)"
    )
    parser.add_argument(
        "--prediction",
        required=True,
        help="Path to predictions file (one prediction per line)"
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to target/reference file (one target per line)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output file (default: <prediction_file>_results.txt)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.prediction):
        print(f"Error: Predictions file not found: {args.prediction}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.target):
        print(f"Error: Target file not found: {args.target}", file=sys.stderr)
        sys.exit(1)
    
    # Load files - preserve empty lines to maintain alignment
    with open(args.prediction, "r", encoding="utf-8") as f:
        preds = [line.rstrip('\n') for line in f]
    
    with open(args.target, "r", encoding="utf-8") as f:
        refs = [line.rstrip('\n') for line in f]
    
    # Check that line counts match
    if len(preds) != len(refs):
        print(f"Error: Line count mismatch: {len(preds)} predictions vs {len(refs)} references", file=sys.stderr)
        print(f"  Prediction file: {args.prediction}", file=sys.stderr)
        print(f"  Target file: {args.target}", file=sys.stderr)
        sys.exit(1)
    
    # Filter out pairs where prediction is empty (to match inference script behavior)
    # But maintain alignment by keeping track of which lines were filtered
    filtered_preds = []
    filtered_refs = []
    for pred, ref in zip(preds, refs):
        if pred.strip():  # Only include non-empty predictions
            filtered_preds.append(pred)
            filtered_refs.append(ref)
    
    preds = filtered_preds
    refs = filtered_refs
    
    # Sanity check after filtering
    if len(preds) != len(refs):
        print(f"Error: Line count mismatch after filtering: {len(preds)} predictions vs {len(refs)} references", file=sys.stderr)
        sys.exit(1)
    
    # Metadata
    sentence_count = len(preds)
    word_count_preds = count_words(preds)
    word_count_refs = count_words(refs)
    
    # Calculate metrics
    print("Calculating metrics...")
    
    avg_lev_per_valid, avg_lev_per_line, avg_lev_per_pred = calculate_levenshtein_distance(preds, refs)
    character_accuracy = calculate_character_accuracy(preds, refs)
    gold_count = calculate_gold(preds, refs)
    rouge_l, sacrebleu = calculate_rouge_and_bleu(preds, refs)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        pred_basename = os.path.basename(args.prediction)
        pred_name = os.path.splitext(pred_basename)[0]
        output_file = f"{pred_name}_results.txt"
    
    # -------------------------------
    # PRINT RESULTS
    # -------------------------------
    print("========================================")
    print(" RAMSES Transliteration Evaluation")
    print("========================================")
    print(f"Predictions file: {args.prediction}")
    print(f"Reference file:   {args.target}")
    print("----------------------------------------")
    print(f"Sentence count: {sentence_count}")
    print(f"Word count (pred): {word_count_preds}")
    print(f"Word count (ref):  {word_count_refs}")
    print("----------------------------------------")
    print(f"GOLD (perfect matches): {gold_count} ({100*gold_count/sentence_count:.2f}%)")
    print(f"Character accuracy: {character_accuracy:.7f}")
    print(f"Average Levenshtein distance per valid token: {avg_lev_per_valid:.7f}")
    if rouge_l is not None:
        print(f"ROUGE-L   (token-level): {rouge_l:.2f}%")
    if sacrebleu is not None:
        print(f"SacreBLEU (token-level): {sacrebleu:.2f}")
    print("========================================")
    
    # -------------------------------
    # SAVE RESULTS TO FILE
    # -------------------------------
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("RAMSES Transliteration Metrics\n")
        out.write("=================================\n")
        out.write(f"Predictions file: {args.prediction}\n")
        out.write(f"Reference file:   {args.target}\n")
        out.write("---------------------------------\n")
        out.write(f"Sentence count: {sentence_count}\n")
        out.write(f"Word count (pred): {word_count_preds}\n")
        out.write(f"Word count (ref):  {word_count_refs}\n")
        out.write("---------------------------------\n")
        out.write(f"GOLD (perfect matches): {gold_count} ({100*gold_count/sentence_count:.4f}%)\n")
        out.write(f"Character accuracy: {character_accuracy:.7f}\n")
        out.write(f"Average Levenshtein distance per valid token: {avg_lev_per_valid:.7f}\n")
        if rouge_l is not None:
            out.write(f"ROUGE-L   (token-level): {rouge_l:.4f}\n")
        if sacrebleu is not None:
            out.write(f"SacreBLEU (token-level): {sacrebleu:.4f}\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

