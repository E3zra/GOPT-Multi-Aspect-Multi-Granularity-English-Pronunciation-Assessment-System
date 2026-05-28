#!/usr/bin/env python3
"""
Generate text-phone file for Kaldi GOP extraction.
Reads valid phones from lang_nosp/phones.txt and generates
text-phone entries using only valid position-dependent phones.

Usage: python3 generate_text_phone.py <lexicon_file> <transcript> <output_file> <utt_id>
"""
import sys
import os
import re

def load_lexicon(lexicon_path):
    """Load lexicon: word -> list of phones"""
    lexicon = {}
    with open(lexicon_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                phones = parts[1:]
                lexicon[word] = phones
    return lexicon

def load_valid_phones(phones_path):
    """Load set of valid phone names from phones.txt"""
    valid = set()
    with open(phones_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                valid.add(parts[0])
    return valid

def get_position_phones(base_phones, valid_phones):
    """Add word-position markers to phone sequence.
    Only adds markers if they exist in valid_phones.
    Returns list of position-dependent phone names.
    """
    n = len(base_phones)
    result = []
    
    for i, ph in enumerate(base_phones):
        if n == 1:
            candidates = [f"{ph}_S", ph, f"{ph}_B", f"{ph}_E"]
        elif i == 0:
            candidates = [f"{ph}_B", ph]
        elif i == n - 1:
            candidates = [f"{ph}_E", ph]
        else:
            candidates = [f"{ph}_I", ph]
        
        # Use first valid candidate
        chosen = None
        for c in candidates:
            if c in valid_phones:
                chosen = c
                break
        
        if chosen is None:
            # Fallback: use base phone if it's valid
            if ph in valid_phones:
                chosen = ph
            else:
                # Last resort: use first phone in valid_phones as OOV marker
                # or skip
                print(f"Warning: No valid phone for {ph} in position {i}, using SPN")
                chosen = "SPN" if "SPN" in valid_phones else "SIL"
        
        result.append(chosen)
    
    return result

def main():
    if len(sys.argv) < 5:
        print("Usage: python3 generate_text_phone.py <lexicon_file> <transcript> <output_file> <utt_id>")
        sys.exit(1)
    
    lexicon_path = sys.argv[1]
    transcript = sys.argv[2].upper()
    output_path = sys.argv[3]
    utt_id = sys.argv[4]
    
    # Phones.txt path (derived from lexicon location)
    kaldi_dir = os.path.dirname(os.path.dirname(os.path.dirname(lexicon_path)))
    phones_path = os.path.join(kaldi_dir, "data", "lang_nosp", "phones.txt")
    
    # If phones.txt doesn't exist yet, try alternate location
    if not os.path.exists(phones_path):
        # We'll generate without position markers and rely on prepare_lang.sh
        # to create the phones, then this will be called again after lang prep
        print(f"Warning: {phones_path} not found, generating without position markers")
        valid_phones = None
    else:
        valid_phones = load_valid_phones(phones_path)
        print(f"Loaded {len(valid_phones)} valid phones")
    
    # Load lexicon
    lexicon = load_lexicon(lexicon_path)
    print(f"Loaded {len(lexicon)} words from lexicon")
    
    # Generate phone sequence
    words = transcript.strip().split()
    phone_seq = ["SIL"]
    
    for word in words:
        if word in lexicon:
            base_phones = lexicon[word]
        else:
            print(f"Warning: OOV word '{word}' not in lexicon, using SPN")
            base_phones = ["SPN"]
        
        if valid_phones is not None:
            pos_phones = get_position_phones(base_phones, valid_phones)
        else:
            # No position markers available yet
            pos_phones = base_phones
        
        phone_seq.extend(pos_phones)
    
    phone_seq.append("SIL")
    
    # Write text-phone file (one entry per phone)
    with open(output_path, 'w') as f:
        for idx, phone in enumerate(phone_seq):
            f.write(f"{utt_id}.{idx} {phone}\n")
    
    print(f"Generated {len(phone_seq)} phones: {' '.join(phone_seq[:10])}...")
    print(f"Written to {output_path}")

if __name__ == "__main__":
    main()