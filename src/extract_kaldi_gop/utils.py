# -*- coding: utf-8 -*-
# @Time    : 8/20/21 4:02 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : utils.py

# Utility functions for GOP feature extraction

import json
import os


def load_phone_symbol_table(phone_symbol_table_path):
    """
    Load phone symbol table mapping phone symbols to integer IDs
    
    Args:
        phone_symbol_table_path: Path to phones.txt or phones-pure.txt
        
    Returns:
        Tuple of (phone_sym2int, phone_int2sym) dictionaries
    """
    if not os.path.exists(phone_symbol_table_path):
        print(f"Warning: Phone symbol table not found at {phone_symbol_table_path}")
        return {}, {}
    
    phone_sym2int = {}
    phone_int2sym = {}
    
    with open(phone_symbol_table_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                phone_symbol = parts[0]
                phone_id = int(parts[1])
                phone_sym2int[phone_symbol] = phone_id
                phone_int2sym[phone_id] = phone_symbol
    
    print(f"Loaded {len(phone_sym2int)} phone symbols from {phone_symbol_table_path}")
    return phone_sym2int, phone_int2sym


def load_human_scores(human_scoring_json_path, floor=0.1):
    """
    Load human pronunciation scores from JSON file
    
    Args:
        human_scoring_json_path: Path to scores.json file
        floor: Minimum score floor value (default 0.1)
        
    Returns:
        Tuple of (score_of, phone_of) dictionaries
        - score_of: Maps utterance.phone_idx -> pronunciation score
        - phone_of: Maps utterance.phone_idx -> phone symbol
    """
    if not os.path.exists(human_scoring_json_path):
        print(f"Warning: Human scoring JSON not found at {human_scoring_json_path}")
        return {}, {}
    
    with open(human_scoring_json_path, 'r') as f:
        info = json.loads(f.read())
    
    score_of = {}
    phone_of = {}
    
    # Parse the JSON structure
    # Expected format: 
    # {
    #   "utterance_id": {
    #     "words": [
    #       {
    #         "text": "WORD",
    #         "phones": ["F", "OW", "N"],
    #         "phones-accuracy": [2, 1.8, 2]
    #       }
    #     ]
    #   }
    # }
    
    for utt_id in info:
        if 'words' not in info[utt_id]:
            continue
            
        phone_num = 0
        for word in info[utt_id]['words']:
            if 'phones' not in word or 'phones-accuracy' not in word:
                continue
                
            phones = word['phones']
            phones_accuracy = word['phones-accuracy']
            
            for i, phone in enumerate(phones):
                key = f'{utt_id}.{phone_num}'
                
                # Apply floor to scores
                score = phones_accuracy[i] if i < len(phones_accuracy) else floor
                score = max(score, floor)
                
                score_of[key] = score
                phone_of[key] = phone
                phone_num += 1
    
    print(f"Loaded scores for {len(score_of)} phones from {human_scoring_json_path}")
    return score_of, phone_of

