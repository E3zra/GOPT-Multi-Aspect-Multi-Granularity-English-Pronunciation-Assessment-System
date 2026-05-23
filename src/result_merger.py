#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result Merging Module

This module merges results from multiple audio segments into a unified
assessment result with proper weighting and concatenation.
"""

from typing import List, Dict, Any
import numpy as np


class ResultMerger:
    """
    Merge pronunciation assessment results from multiple segments.
    
    Handles:
    - Weighted averaging of utterance-level scores
    - Concatenation of word-level scores
    - Concatenation of phone-level scores
    - Preservation of segment metadata
    """
    
    def __init__(self):
        """Initialize the result merger."""
        pass
    
    def merge_results(
        self,
        segment_results: List[Dict[str, Any]],
        text_segments: List = None
    ) -> Dict[str, Any]:
        """
        Merge results from multiple segments.
        
        Strategy:
        - Utterance scores: weighted average by phoneme count
        - Word scores: concatenate in order, renumber word_ids
        - Phone scores: concatenate in order
        
        Args:
            segment_results: List of result dictionaries from each segment
            text_segments: Optional list of TextSegment objects for metadata
            
        Returns:
            Merged result dictionary
        """
        if not segment_results:
            raise ValueError("No results to merge")
        
        if len(segment_results) == 1:
            # Single segment, return as-is but add metadata
            result = segment_results[0].copy()
            result['segmentation_info'] = {
                'num_segments': 1,
                'segments': [{
                    'phoneme_count': len(result.get('phone_scores', [])),
                    'word_count': len(result.get('word_scores', []))
                }]
            }
            return result
        
        # Multiple segments - need to merge
        merged_result = {
            'utterance_scores': {},
            'word_scores': [],
            'phone_scores': [],
            'segmentation_info': {
                'num_segments': len(segment_results),
                'segments': []
            }
        }
        
        # 1. Merge utterance-level scores (weighted by phoneme count)
        merged_result['utterance_scores'] = self._merge_utterance_scores(
            segment_results
        )
        
        # 2. Concatenate word-level scores
        merged_result['word_scores'] = self._merge_word_scores(
            segment_results
        )
        
        # 3. Concatenate phone-level scores
        merged_result['phone_scores'] = self._merge_phone_scores(
            segment_results
        )
        
        # 4. Add segmentation metadata
        for i, result in enumerate(segment_results):
            segment_info = {
                'segment_index': i,
                'phoneme_count': len(result.get('phone_scores', [])),
                'word_count': len(result.get('word_scores', [])),
            }
            
            # Add text segment info if available
            if text_segments and i < len(text_segments):
                segment_info['text'] = text_segments[i].text
                segment_info['start_word_idx'] = text_segments[i].start_word_idx
                segment_info['end_word_idx'] = text_segments[i].end_word_idx
            
            merged_result['segmentation_info']['segments'].append(segment_info)
        
        return merged_result
    
    def _merge_utterance_scores(
        self,
        segment_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Merge utterance-level scores using weighted average.
        
        Weight: number of valid phonemes in each segment
        
        Args:
            segment_results: List of result dictionaries
            
        Returns:
            Merged utterance scores dictionary
        """
        # Calculate weights (number of valid phonemes per segment)
        weights = []
        for result in segment_results:
            phone_scores = result.get('phone_scores', [])
            # Count valid phones (score >= 0)
            valid_phones = sum(1 for score in phone_scores if score >= 0)
            weights.append(valid_phones)
        
        total_weight = sum(weights)
        
        if total_weight == 0:
            # No valid data, return zeros
            return {
                'accuracy': 0.0,
                'completeness': 0.0,
                'fluency': 0.0,
                'prosodic': 0.0,
                'total': 0.0
            }
        
        # Weighted average for each aspect
        merged_scores = {}
        aspect_names = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']
        
        for aspect in aspect_names:
            weighted_sum = 0.0
            for result, weight in zip(segment_results, weights):
                utterance_scores = result.get('utterance_scores', {})
                score = utterance_scores.get(aspect, 0.0)
                weighted_sum += score * weight
            
            merged_scores[aspect] = weighted_sum / total_weight
        
        return merged_scores
    
    def _merge_word_scores(
        self,
        segment_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Concatenate word-level scores from all segments.
        
        Renumbers word_id to be sequential across all segments.
        
        Args:
            segment_results: List of result dictionaries
            
        Returns:
            Concatenated list of word score dictionaries
        """
        merged_words = []
        current_word_id = 0
        
        for result in segment_results:
            word_scores = result.get('word_scores', [])
            
            for word_score in word_scores:
                # Create new word score entry with renumbered id
                merged_word = word_score.copy()
                merged_word['word_id'] = current_word_id
                merged_words.append(merged_word)
                current_word_id += 1
        
        return merged_words
    
    def _merge_phone_scores(
        self,
        segment_results: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Concatenate phone-level scores from all segments.
        
        Args:
            segment_results: List of result dictionaries
            
        Returns:
            Concatenated list of phone scores
        """
        merged_phones = []
        
        for result in segment_results:
            phone_scores = result.get('phone_scores', [])
            merged_phones.extend(phone_scores)
        
        return merged_phones
    
    def get_segment_statistics(self, merged_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract statistics about the merged result.
        
        Args:
            merged_result: Merged result dictionary
            
        Returns:
            Statistics dictionary
        """
        seg_info = merged_result.get('segmentation_info', {})
        
        total_words = len(merged_result.get('word_scores', []))
        total_phones = len(merged_result.get('phone_scores', []))
        
        # Count valid phones
        valid_phones = sum(
            1 for score in merged_result.get('phone_scores', [])
            if score >= 0
        )
        
        return {
            'num_segments': seg_info.get('num_segments', 1),
            'total_words': total_words,
            'total_phonemes': total_phones,
            'valid_phonemes': valid_phones,
            'segments': seg_info.get('segments', [])
        }


def main():
    """Demo usage of ResultMerger."""
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    print("=" * 70)
    print("Result Merging Demo")
    print("=" * 70)
    
    # Example: simulate two segment results
    segment1 = {
        'utterance_scores': {
            'accuracy': 0.75,
            'completeness': 1.90,
            'fluency': 0.95,
            'prosodic': 0.85,
            'total': 0.92
        },
        'word_scores': [
            {'word_id': 0, 'word_text': 'HI', 'accuracy': 1.66, 'stress': 1.96, 'total': 1.73, 'phone_count': 2},
            {'word_id': 1, 'word_text': 'I', 'accuracy': 1.66, 'stress': 1.98, 'total': 1.75, 'phone_count': 1},
            {'word_id': 2, 'word_text': 'AM', 'accuracy': 1.39, 'stress': 1.83, 'total': 1.46, 'phone_count': 3},
        ],
        'phone_scores': [1.92, 1.90, 1.89, 1.70, 1.54, 1.38]  # 6 phones
    }
    
    segment2 = {
        'utterance_scores': {
            'accuracy': 0.80,
            'completeness': 1.85,
            'fluency': 1.00,
            'prosodic': 0.90,
            'total': 0.96
        },
        'word_scores': [
            {'word_id': 0, 'word_text': 'AND', 'accuracy': 1.39, 'stress': 1.86, 'total': 1.46, 'phone_count': 3},
            {'word_id': 1, 'word_text': 'HOW', 'accuracy': 1.64, 'stress': 1.93, 'total': 1.66, 'phone_count': 2},
        ],
        'phone_scores': [1.39, 1.66, 1.60, 1.88, 1.65]  # 5 phones
    }
    
    merger = ResultMerger()
    
    print("\nSegment 1:")
    print(f"  Words: {len(segment1['word_scores'])}")
    print(f"  Phones: {len(segment1['phone_scores'])}")
    print(f"  Total score: {segment1['utterance_scores']['total']:.3f}")
    
    print("\nSegment 2:")
    print(f"  Words: {len(segment2['word_scores'])}")
    print(f"  Phones: {len(segment2['phone_scores'])}")
    print(f"  Total score: {segment2['utterance_scores']['total']:.3f}")
    
    # Merge results
    merged = merger.merge_results([segment1, segment2])
    
    print("\n" + "-" * 70)
    print("Merged Result:")
    print("-" * 70)
    
    print(f"\nUtterance Scores (weighted average):")
    for aspect, score in merged['utterance_scores'].items():
        print(f"  {aspect:15s}: {score:.3f}")
    
    print(f"\nWord Scores: {len(merged['word_scores'])} words")
    for word in merged['word_scores']:
        print(f"  {word['word_id']:2d}. {word['word_text']:10s} Total: {word['total']:.2f}")
    
    print(f"\nPhone Scores: {len(merged['phone_scores'])} phonemes")
    
    # Statistics
    stats = merger.get_segment_statistics(merged)
    print(f"\nStatistics:")
    print(f"  Segments: {stats['num_segments']}")
    print(f"  Total words: {stats['total_words']}")
    print(f"  Total phonemes: {stats['total_phonemes']}")
    print(f"  Valid phonemes: {stats['valid_phonemes']}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

