#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G2P Helper Module for GOPT

Provides grapheme-to-phoneme conversion for out-of-vocabulary (OOV) words
using the g2p-en library. This ensures that words not in the LibriSpeech
lexicon can still be processed by generating phoneme sequences automatically.

Usage:
    from src.utils.g2p_helper import G2PConverter
    
    g2p = G2PConverter()
    phonemes = g2p.convert("STUDENTS")
    # Returns: ['S', 'T', 'UW', 'D', 'AH', 'N', 'T', 'S']
"""

import re
from typing import List, Optional


class G2PConverter:
    """
    Grapheme-to-Phoneme converter for English words.
    
    Uses g2p-en library to convert words to ARPAbet phoneme sequences,
    compatible with LibriSpeech lexicon format.
    """
    
    def __init__(self, remove_stress: bool = True):
        """
        Initialize the G2P converter.
        
        Args:
            remove_stress: If True, remove stress markers (0,1,2) from phonemes
                          to match LibriSpeech lexicon format which may not
                          consistently use stress markers.
        """
        self.remove_stress = remove_stress
        self._g2p = None
        self._initialize_g2p()
    
    def _initialize_g2p(self):
        """Lazy initialization of g2p-en library."""
        try:
            from g2p_en import G2p
            self._g2p = G2p()
            print("G2P converter initialized successfully")
        except ImportError as e:
            print("ERROR: g2p-en library not installed.")
            print("Please install it with: pip install g2p-en")
            raise e
        except Exception as e:
            print(f"ERROR: Failed to initialize G2P converter: {e}")
            raise e
    
    def _normalize_phoneme(self, phoneme: str) -> str:
        """
        Normalize phoneme to match LibriSpeech format.
        
        Args:
            phoneme: Raw phoneme from g2p-en (may include stress markers)
            
        Returns:
            Normalized phoneme without stress markers if remove_stress=True
        """
        if self.remove_stress:
            # Remove stress markers (0, 1, 2) from vowels
            phoneme = re.sub(r'[012]$', '', phoneme)
        return phoneme.upper()
    
    def convert(self, word: str) -> List[str]:
        """
        Convert a word to a list of phonemes.
        
        Args:
            word: English word (case-insensitive)
            
        Returns:
            List of phoneme strings in ARPAbet format
            Returns empty list if conversion fails
            
        Example:
            >>> g2p = G2PConverter()
            >>> g2p.convert("STUDENTS")
            ['S', 'T', 'UW', 'D', 'AH', 'N', 'T', 'S']
        """
        if not word or not isinstance(word, str):
            return []
        
        # Clean the word
        word = word.strip().upper()
        
        # Handle empty word
        if not word:
            return []
        
        try:
            # Get phonemes from g2p-en
            # g2p-en returns phonemes with possible stress markers
            raw_phonemes = self._g2p(word)
            
            # Filter out non-phoneme symbols (like spaces, punctuation)
            # and normalize phonemes
            phonemes = []
            for p in raw_phonemes:
                # Skip non-alphabetic symbols (spaces, punctuation, etc.)
                if not p or not any(c.isalpha() for c in p):
                    continue
                normalized = self._normalize_phoneme(p)
                if normalized:  # Only add non-empty phonemes
                    phonemes.append(normalized)
            
            return phonemes
            
        except Exception as e:
            print(f"ERROR: Failed to convert word '{word}' to phonemes: {e}")
            return []
    
    def convert_with_positions(self, word: str) -> List[str]:
        """
        Convert a word to phonemes with position markers (_B, _I, _E, _S).
        
        Args:
            word: English word
            
        Returns:
            List of phonemes with position suffixes
            
        Example:
            >>> g2p = G2PConverter()
            >>> g2p.convert_with_positions("STUDENTS")
            ['S_B', 'T_I', 'UW_I', 'D_I', 'AH_I', 'N_I', 'T_I', 'S_E']
        """
        phonemes = self.convert(word)
        
        if not phonemes:
            return []
        
        if len(phonemes) == 1:
            # Single phoneme word gets _S (singleton)
            return [phonemes[0] + '_S']
        else:
            # Multi-phoneme word: _B (begin), _I (internal), _E (end)
            result = [phonemes[0] + '_B']
            for p in phonemes[1:-1]:
                result.append(p + '_I')
            result.append(phonemes[-1] + '_E')
            return result
    
    def batch_convert(self, words: List[str]) -> dict:
        """
        Convert multiple words to phonemes.
        
        Args:
            words: List of words to convert
            
        Returns:
            Dictionary mapping word -> list of phonemes
            
        Example:
            >>> g2p = G2PConverter()
            >>> g2p.batch_convert(["STUDENTS", "GORDON"])
            {'STUDENTS': ['S', 'T', 'UW', 'D', 'AH', 'N', 'T', 'S'],
             'GORDON': ['G', 'AO', 'R', 'D', 'AH', 'N']}
        """
        result = {}
        for word in words:
            if word:
                result[word.upper()] = self.convert(word)
        return result


def generate_phonemes_g2p(word: str, remove_stress: bool = True) -> List[str]:
    """
    Convenience function to convert a single word to phonemes.
    
    This is the main function used by process_custom_audio.sh
    
    Args:
        word: Word to convert
        remove_stress: Whether to remove stress markers from vowels
        
    Returns:
        List of phoneme strings
        
    Example:
        >>> from src.utils.g2p_helper import generate_phonemes_g2p
        >>> generate_phonemes_g2p("STUDENTS")
        ['S', 'T', 'UW', 'D', 'AH', 'N', 'T', 'S']
    """
    converter = G2PConverter(remove_stress=remove_stress)
    return converter.convert(word)


# Singleton instance for reuse
_global_converter = None


def get_g2p_converter() -> G2PConverter:
    """
    Get a global G2P converter instance (singleton pattern).
    
    This avoids reinitializing the g2p-en model multiple times.
    
    Returns:
        Global G2PConverter instance
    """
    global _global_converter
    if _global_converter is None:
        _global_converter = G2PConverter()
    return _global_converter


if __name__ == '__main__':
    # Demo usage
    print("="*70)
    print("G2P Helper Module - Demo")
    print("="*70)
    
    # Test words
    test_words = [
        "STUDENTS",
        "GORDON",
        "PERFORMANCE",
        "HEALTH",
        "MORNING",
        "CLASSES",
        "ESSAY",
        "AFFECT"
    ]
    
    print("\nTesting G2P conversion:")
    print("-"*70)
    
    g2p = G2PConverter()
    
    for word in test_words:
        phonemes = g2p.convert(word)
        phonemes_with_pos = g2p.convert_with_positions(word)
        print(f"\n{word:15s}")
        print(f"  Phonemes:      {' '.join(phonemes)}")
        print(f"  With positions: {' '.join(phonemes_with_pos)}")
    
    print("\n" + "="*70)
    print("Demo complete!")

