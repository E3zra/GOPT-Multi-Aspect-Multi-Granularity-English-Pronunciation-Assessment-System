#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Segmentation Module for Long Transcripts

This module intelligently segments long transcripts into smaller chunks
that fit within the GOPT model's 50-phoneme limit.
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TextSegment:
    """Represents a text segment with metadata."""
    text: str
    words: List[str]
    start_word_idx: int
    end_word_idx: int
    phoneme_count: int
    word_count: int
    
    def __repr__(self):
        return f"TextSegment({self.word_count} words, ~{self.phoneme_count} phonemes)"


class TextSegmenter:
    """
    Intelligently segment transcripts based on phoneme counts.
    
    Uses LibriSpeech lexicon to estimate phoneme counts and segments
    text at word boundaries to keep each segment under the limit.
    """
    
    def __init__(self, max_phonemes: int = 45, docker_container: str = "gopt-pipeline"):
        """
        Initialize the text segmenter.
        
        Args:
            max_phonemes: Maximum phonemes per segment (default: 45, leaving margin)
            docker_container: Name of Docker container with lexicon
        """
        self.max_phonemes = max_phonemes
        self.docker_container = docker_container
        self.lexicon: Optional[Dict[str, List[str]]] = None
        
    def load_lexicon(self) -> Dict[str, List[str]]:
        """
        Load LibriSpeech lexicon from Docker container.
        
        Returns:
            Dictionary mapping words to phoneme lists
        """
        if self.lexicon is not None:
            return self.lexicon
        
        try:
            result = subprocess.run(
                ["docker", "exec", self.docker_container, "cat",
                 "/opt/kaldi/egs/gop_speechocean762/s5/librispeech-lexicon.txt"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lexicon = {}
                for line in result.stdout.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = parts[0].upper()
                        phonemes = parts[1:]
                        lexicon[word] = phonemes
                
                self.lexicon = lexicon
                return lexicon
            else:
                print(f"Warning: Could not load lexicon from Docker container")
                return {}
                
        except Exception as e:
            print(f"Warning: Error loading lexicon: {e}")
            return {}
    
    def estimate_phoneme_count(self, word: str) -> int:
        """
        Estimate the number of phonemes in a word.
        
        Args:
            word: Word to estimate (should be uppercase)
            
        Returns:
            Estimated phoneme count
        """
        word = word.upper()
        
        # Try to get from lexicon first
        if self.lexicon and word in self.lexicon:
            return len(self.lexicon[word])
        
        # Fallback: empirical formula
        # English words average ~0.75 phonemes per letter
        # Common patterns:
        # - Short words (2-3 letters): usually 2-3 phonemes
        # - Medium words (4-8 letters): 3-6 phonemes
        # - Long words (9+ letters): 7-11 phonemes
        
        length = len(word)
        if length <= 2:
            return 2
        elif length <= 4:
            return 3
        elif length <= 6:
            return int(length * 0.75)
        else:
            return int(length * 0.7)
    
    def segment_transcript(self, transcript: str) -> List[TextSegment]:
        """
        Intelligently segment transcript into chunks.
        
        Strategy:
        - Greedy algorithm: fill each segment up to max_phonemes
        - Never split words
        - Handle long words (>max_phonemes) specially
        
        Args:
            transcript: Full transcript text (uppercase, space-separated)
            
        Returns:
            List of TextSegment objects
        """
        # Load lexicon if not already loaded
        if self.lexicon is None:
            self.load_lexicon()
        
        # Parse words
        words = transcript.upper().strip().split()
        
        if not words:
            return []
        
        # Calculate total phonemes for info
        total_phonemes = sum(self.estimate_phoneme_count(w) for w in words)
        
        # If fits in one segment, return directly
        if total_phonemes <= self.max_phonemes:
            return [TextSegment(
                text=transcript.upper(),
                words=words,
                start_word_idx=0,
                end_word_idx=len(words) - 1,
                phoneme_count=total_phonemes,
                word_count=len(words)
            )]
        
        # Need to segment
        segments = []
        current_words = []
        current_phonemes = 0
        start_idx = 0
        
        for i, word in enumerate(words):
            word_phonemes = self.estimate_phoneme_count(word)
            
            # Handle extremely long single words
            if word_phonemes > self.max_phonemes:
                # If we have accumulated words, save them first
                if current_words:
                    segments.append(TextSegment(
                        text=' '.join(current_words),
                        words=current_words.copy(),
                        start_word_idx=start_idx,
                        end_word_idx=i - 1,
                        phoneme_count=current_phonemes,
                        word_count=len(current_words)
                    ))
                    current_words = []
                    current_phonemes = 0
                    start_idx = i
                
                # Put long word in its own segment (will be truncated by model)
                segments.append(TextSegment(
                    text=word,
                    words=[word],
                    start_word_idx=i,
                    end_word_idx=i,
                    phoneme_count=word_phonemes,
                    word_count=1
                ))
                start_idx = i + 1
                continue
            
            # Check if adding this word exceeds limit
            if current_phonemes + word_phonemes <= self.max_phonemes:
                # Add to current segment
                current_words.append(word)
                current_phonemes += word_phonemes
            else:
                # Save current segment and start new one
                if current_words:
                    segments.append(TextSegment(
                        text=' '.join(current_words),
                        words=current_words.copy(),
                        start_word_idx=start_idx,
                        end_word_idx=i - 1,
                        phoneme_count=current_phonemes,
                        word_count=len(current_words)
                    ))
                
                # Start new segment with current word
                current_words = [word]
                current_phonemes = word_phonemes
                start_idx = i
        
        # Add final segment if any words remain
        if current_words:
            segments.append(TextSegment(
                text=' '.join(current_words),
                words=current_words.copy(),
                start_word_idx=start_idx,
                end_word_idx=len(words) - 1,
                phoneme_count=current_phonemes,
                word_count=len(current_words)
            ))
        
        return segments
    
    def get_segmentation_info(self, transcript: str) -> Dict:
        """
        Get segmentation information without actually segmenting.
        
        Args:
            transcript: Transcript to analyze
            
        Returns:
            Dictionary with segmentation statistics
        """
        segments = self.segment_transcript(transcript)
        
        total_words = sum(seg.word_count for seg in segments)
        total_phonemes = sum(seg.phoneme_count for seg in segments)
        
        return {
            'total_words': total_words,
            'total_phonemes': total_phonemes,
            'num_segments': len(segments),
            'needs_segmentation': len(segments) > 1,
            'segments': [
                {
                    'words': seg.word_count,
                    'phonemes': seg.phoneme_count,
                    'text': seg.text
                }
                for seg in segments
            ]
        }


def main():
    """Demo usage of TextSegmenter."""
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # Test cases
    test_transcripts = [
        "HI I AM GORDON",  # Short
        "HI I AM GORDON MY ESSAY IS ABOUT EARLY MORNING CLASSES AND HOW THEY AFFECT STUDENTS PERFORMANCE AND HEALTH",  # Long
    ]
    
    segmenter = TextSegmenter(max_phonemes=45)
    
    print("=" * 70)
    print("Text Segmentation Demo")
    print("=" * 70)
    print(f"Max phonemes per segment: {segmenter.max_phonemes}\n")
    
    for transcript in test_transcripts:
        print(f"\nTranscript: {transcript[:60]}...")
        print("-" * 70)
        
        segments = segmenter.segment_transcript(transcript)
        
        print(f"Total segments: {len(segments)}")
        for i, seg in enumerate(segments, 1):
            print(f"\nSegment {i}:")
            print(f"  Words: {seg.word_count} ({seg.start_word_idx}-{seg.end_word_idx})")
            print(f"  Phonemes: ~{seg.phoneme_count}")
            print(f"  Text: {seg.text}")
        
        print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

