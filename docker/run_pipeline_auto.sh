################################################################################
# GOPT Full Pipeline Automation Script (Docker) - FULLY AUTOMATED
# 
# This script automates the ENTIRE GOPT pronunciation assessment pipeline:
#   1. Prepare Kaldi data directory
#   2. Generate lexicon and phone transcripts for custom audio
#   3. Extract MFCC features
#   4. Extract i-vectors
#   5. Compute nnet3 output (log-likelihoods)
#   6. Prepare lang (dict, phone mapping)
#   7. Make align graphs and align
#   8. Compute GOP features
#   9. Extract GOP features to CSV (custom, no human scores needed)
#  10. Convert to sequence format
#  11. Run GOPT inference
#
# Usage:
#   ./run_pipeline_auto.sh <audio_file> <transcript> [dataset_name]
#
# Example:
#   ./run_pipeline_auto.sh /workspace/audio_input/test.wav "HELLO WORLD" my_audio
################################################################################

# Don't use set -e globally - some commands legitimately return non-zero
# Instead, handle errors explicitly in critical sections
set +e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GOPT_ROOT="/workspace/gopt"
KALDI_ROOT="/opt/kaldi"
KALDI_GOP_DIR="$KALDI_ROOT/egs/gop_speechocean762/s5"
LIBRISPEECH_MODEL_DIR="/workspace/models/librispeech"

# LibriSpeech model paths
CHAIN_MODEL="$LIBRISPEECH_MODEL_DIR/exp/chain_cleaned/tdnn_1d_sp"
IVECTOR_EXTRACTOR="$LIBRISPEECH_MODEL_DIR/exp/nnet3_cleaned/extractor"
LANG_DIR="$LIBRISPEECH_MODEL_DIR/data/lang_test_tgsmall"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found!"
        exit 1
    fi
    print_success "Python3 found: $(python3 --version)"
    
    # Check PyTorch
    if ! python3 -c "import torch" 2>/dev/null; then
        print_error "PyTorch not installed!"
        exit 1
    fi
    print_success "PyTorch found"
    
    # Check Kaldi
    if [ ! -d "$KALDI_ROOT" ]; then
        print_error "Kaldi not found at $KALDI_ROOT"
        exit 1
    fi
    print_success "Kaldi found at $KALDI_ROOT"
    
    # Check LibriSpeech models
    if [ ! -d "$CHAIN_MODEL" ]; then
        print_error "Chain model not found at $CHAIN_MODEL"
        exit 1
    fi
    print_success "Chain model found"
    
    if [ ! -d "$IVECTOR_EXTRACTOR" ]; then
        print_error "i-vector extractor not found at $IVECTOR_EXTRACTOR"
        exit 1
    fi
    print_success "i-vector extractor found"
    
    if [ ! -d "$LANG_DIR" ]; then
        print_error "Lang dir not found at $LANG_DIR"
        exit 1
    fi
    print_success "Lang dir found"
    
    # Check pretrained GOPT model
    if [ ! -f "$GOPT_ROOT/pretrained_models/gopt_librispeech/best_audio_model.pth" ]; then
        print_error "GOPT pretrained model not found!"
        exit 1
    fi
    print_success "GOPT pretrained model found"
}

# Function to validate audio file
validate_audio() {
    local audio_file=$1
    
    print_info "Validating audio file: $audio_file"
    
    if [ ! -f "$audio_file" ]; then
        print_error "Audio file not found: $audio_file"
        exit 1
    fi
    
    # Check if it's a WAV file
    if [[ ! "$audio_file" =~ \.wav$ ]]; then
        print_error "Audio file is not .wav format. Please convert to 16kHz mono WAV first."
        exit 1
    fi
    
    print_success "Audio file validated"
}

# Step 1: Prepare Kaldi data directory
prepare_kaldi_data() {
    local audio_file=$1
    local transcript=$2
    local dataset_name=$3
    
    print_info "Step 1: Preparing Kaldi data directory for dataset: $dataset_name"
    
    local data_dir="$KALDI_GOP_DIR/data/$dataset_name"
    local train_dir="$data_dir/train"
    local test_dir="$data_dir/test"
    
    # Create directories
    mkdir -p "$train_dir" "$test_dir"
    
    # Generate IDs
    local utt_id="utt_001"
    local spk_id="spk_001"
    
    # Get absolute path
    audio_file=$(realpath "$audio_file")
    
    # Create Kaldi format files for test set
    echo "$utt_id $audio_file" > "$test_dir/wav.scp"
    echo "$utt_id ${transcript^^}" > "$test_dir/text"  # Convert to uppercase
    echo "$utt_id $spk_id" > "$test_dir/utt2spk"
    echo "$spk_id $utt_id" > "$test_dir/spk2utt"
    
    # Copy to train set (simplified - same data for both)
    cp "$test_dir"/* "$train_dir/"
    
    print_success "Kaldi data directory prepared at: $data_dir"
}

# Step 2: Generate lexicon from transcript
generate_lexicon() {
    local transcript=$2
    
    print_info "Step 2: Generating lexicon from transcript"
    
    # Ensure data/local directory exists
    mkdir -p "$KALDI_GOP_DIR/data/local"
    
    local transcript_upper="${transcript^^}"
    
    # Create lexicon with proper phonemes using G2P or lookup
    # We use a Python script to generate phonemes using g2p_en
    python3 - <<PYEOF
import sys
import os

transcript = "$transcript_upper"
words = transcript.strip().split()
unique_words = sorted(set(words))

# Try to use g2p_en for phoneme generation
try:
    from g2p_en import G2p
    g2p = G2p()
    has_g2p = True
    print("Using g2p_en for phoneme generation")
except ImportError:
    has_g2p = False
    print("g2p_en not available, using CMU dict fallback")

# CMU pronouncing dictionary fallback
cmu_dict = {
    "HI": ["HH", "AY"],
    "I": ["AY"],
    "AM": ["AE", "M"],
    "GORDON": ["G", "AO", "R", "D", "AH", "N"],
    "MY": ["M", "AY"],
    "ESSAY": ["EH", "S", "EY"],
    "IS": ["IH", "Z"],
    "ABOUT": ["AH", "B", "AW", "T"],
    "EARLY": ["ER", "L", "IY"],
    "MORNING": ["M", "AO", "R", "N", "IH", "NG"],
    "CLASS": ["K", "L", "AE", "S"],
    "AND": ["AH", "N", "D"],
    "HOW": ["HH", "AW"],
    "THEY": ["DH", "EY"],
    "AFFECT": ["AE", "F", "EH", "K", "T"],
    "STUDENTS": ["S", "T", "UW", "D", "AH", "N", "T", "S"],
    "PERFORMANCE": ["P", "ER", "F", "AO", "R", "M", "AH", "N", "S"],
    "HEALTH": ["HH", "EH", "L", "TH"],
    "THE": ["DH", "AH"],
    "A": ["AH"],
    "AN": ["AE", "N"],
    "IN": ["IH", "N"],
    "TO": ["T", "UW"],
    "OF": ["AH", "V"],
    "FOR": ["F", "AO", "R"],
    "WITH": ["W", "IH", "DH"],
    "ON": ["AO", "N"],
    "AT": ["AE", "T"],
    "THIS": ["DH", "IH", "S"],
    "BUT": ["B", "AH", "T"],
    "NOT": ["N", "AA", "T"],
    "YOU": ["Y", "UW"],
    "IT": ["IH", "T"],
    "HE": ["HH", "IY"],
    "SHE": ["SH", "IY"],
    "WE": ["W", "IY"],
    "THEY": ["DH", "EY"],
    "WHAT": ["W", "AH", "T"],
    "ALL": ["AO", "L"],
    "WOULD": ["W", "UH", "D"],
    "THERE": ["DH", "EH", "R"],
    "THEIR": ["DH", "EH", "R"],
    "BEEN": ["B", "IH", "N"],
    "MANY": ["M", "EH", "N", "IY"],
    "SOME": ["S", "AH", "M"],
    "SO": ["S", "OW"],
    "THESE": ["DH", "IY", "Z"],
    "HER": ["HH", "ER"],
    "WOULD": ["W", "UH", "D"],
    "MAKE": ["M", "EY", "K"],
    "LIKE": ["L", "AY", "K"],
    "LONG": ["L", "AO", "NG"],
    "LOOK": ["L", "UH", "K"],
    "COME": ["K", "AH", "M"],
    "COULD": ["K", "UH", "D"],
    "SAID": ["S", "EH", "D"],
    "EACH": ["IY", "CH"],
    "WHICH": ["W", "IH", "CH"],
    "DO": ["D", "UW"],
    "THEIR": ["DH", "EH", "R"],
    "TIME": ["T", "AY", "M"],
    "IF": ["IH", "F"],
    "WILL": ["W", "IH", "L"],
    "WAY": ["W", "EY"],
    "ABOUT": ["AH", "B", "AW", "T"],
    "UP": ["AH", "P"],
    "OUT": ["AW", "T"],
    "THEM": ["DH", "EH", "M"],
    "THEN": ["DH", "EH", "N"],
    "THAN": ["DH", "AE", "N"],
    "THEM": ["DH", "EH", "M"],
    "GET": ["G", "EH", "T"],
    "GO": ["G", "OW"],
    "NO": ["N", "OW"],
    "CAN": ["K", "AE", "N"],
    "HAD": ["HH", "AE", "D"],
    "HAS": ["HH", "AE", "Z"],
    "HIS": ["HH", "IH", "Z"],
    "HER": ["HH", "ER"],
    "ONE": ["W", "AH", "N"],
    "OUR": ["AW", "ER"],
    "TWO": ["T", "UW"],
    "JUST": ["JH", "AH", "S", "T"],
    "OVER": ["OW", "V", "ER"],
    "SUCH": ["S", "AH", "CH"],
    "AFTER": ["AE", "F", "T", "ER"],
    "WHEN": ["W", "EH", "N"],
    "WHAT": ["W", "AH", "T"],
    "YOUR": ["Y", "AO", "R"],
    "HOW": ["HH", "AW"],
    "ITS": ["IH", "T", "S"],
    "DID": ["D", "IH", "D"],
    "GET": ["G", "EH", "T"],
    "FROM": ["F", "R", "AH", "M"],
    "OR": ["AO", "R"],
    "HAS": ["HH", "AE", "Z"],
    "MAY": ["M", "EY"],
    "FIRST": ["F", "ER", "S", "T"],
    "VERY": ["V", "EH", "R", "IY"],
    "BE": ["B", "IY"],
    "WAS": ["W", "AH", "Z"],
    "WERE": ["W", "ER"],
    "ARE": ["AA", "R"],
    "BEEN": ["B", "IH", "N"],
    "HAVE": ["HH", "AE", "V"],
    "DO": ["D", "UW"],
    "SAY": ["S", "EY"],
    "HE": ["HH", "IY"],
    "SHE": ["SH", "IY"],
    "WHO": ["HH", "UW"],
    "OIL": ["OY", "L"],
    "SIT": ["S", "IH", "T"],
    "HEALTHY": ["HH", "EH", "L", "TH", "IY"],
    "STUDENT": ["S", "T", "UW", "D", "AH", "N", "T"],
    "PERFORM": ["P", "ER", "F", "AO", "R", "M"],
    "AFFECTS": ["AE", "F", "EH", "K", "T", "S"],
    "MORNINGS": ["M", "AO", "R", "N", "IH", "NG", "Z"],
    "CLASSES": ["K", "L", "AE", "S", "AH", "Z"],
    "AFFECTED": ["AE", "F", "EH", "K", "T", "AH", "D"],
    "PERFORMING": ["P", "ER", "F", "AO", "R", "M", "IH", "NG"],
    "ESSAYS": ["EH", "S", "EY", "Z"],
    "GORDONS": ["G", "AO", "R", "D", "AH", "N", "Z"],
}

# Kaldi-compatible phoneme set (no stress markers, pure phones)
kaldi_phones = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", 
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", 
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", 
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
    "SIL", "SPN"
]

lexicon_lines = []
# Do NOT include SIL or SPN entries - prepare_dict.sh handles silence phones automatically

missing_words = []
for word in unique_words:
    word_upper = word.upper()
    if word_upper in cmu_dict:
        phones = cmu_dict[word_upper]
        lexicon_lines.append(f"{word_upper} {' '.join(phones)}")
    elif has_g2p:
        try:
            phones_raw = g2p(word_upper)
            # Clean up g2p output - remove stress markers and special chars
            phones = []
            for p in phones_raw:
                p_clean = ''.join(c for c in p if c.isalpha())
                if p_clean and p_clean.isalpha():
                    # Convert to uppercase, remove stress markers
                    p_upper = p_clean.upper()
                    # Remove trailing digits (stress markers)
                    p_base = ''.join(c for c in p_upper if c.isalpha())
                    if p_base in kaldi_phones:
                        phones.append(p_base)
                    elif p_base:
                        # Map common variants
                        if p_base == "AX": phones.append("AH")
                        elif p_base == "IX": phones.append("IH")
                        elif p_base == "UX": phones.append("UH")
                        else: phones.append(p_base)
            if phones:
                lexicon_lines.append(f"{word_upper} {' '.join(phones)}")
            else:
                missing_words.append(word_upper)
        except Exception as e:
            print(f"G2P failed for {word_upper}: {e}")
            missing_words.append(word_upper)
    else:
        missing_words.append(word_upper)

if missing_words:
    print(f"Warning: Could not generate phonemes for: {', '.join(missing_words)}")
    print("Adding with fallback phonemes (AH)")
    for w in missing_words:
        # Use a simple fallback - single AH phone is always valid
        # Don't use SPN as it conflicts with silence phones
        lexicon_lines.append(f"{w} AH")

# Write lexicon
lexicon_path = "$KALDI_GOP_DIR/data/local/lexicon.txt"
with open(lexicon_path, 'w') as f:
    for line in sorted(lexicon_lines):
        f.write(line + '\n')

print(f"Lexicon generated with {len(lexicon_lines)} entries at {lexicon_path}")
PYEOF

    print_success "Lexicon generated"
}

# Step 3: Generate text-phone file
generate_text_phone() {
    local transcript=$2
    
    print_info "Step 3: Generating text-phone file (using lexicon-based approach)"
    
    local transcript_upper="${transcript^^}"
    local lexicon_path="$KALDI_GOP_DIR/data/local/lexicon.txt"
    local text_phone_path="$KALDI_GOP_DIR/data/local/text-phone"
    
    # Use the Python helper that reads lexicon and lang_nosp/phones.txt
    python3 "$GOPT_ROOT/docker/generate_text_phone.py" \
        "$lexicon_path" \
        "$transcript_upper" \
        "$text_phone_path" \
        "utt_001"
    
    if [ $? -ne 0 ]; then
        print_error "Failed to generate text-phone file"
        return 1
    fi
    
    print_success "Text-phone file generated"
}

# Step 4: Generate dummy scores.json for custom audio
generate_dummy_scores() {
    local transcript=$2
    
    print_info "Step 4: Generating dummy scores for custom audio"
    
    local transcript_upper="${transcript^^}"
    
    python3 - <<PYEOF
import json

transcript = "$transcript_upper"
words = transcript.strip().split()

# Generate dummy scores (all perfect = 0.0, which means "correct" in GOP scoring)
# The format is: {utterance_id: {phone_idx: score}}
scores = {}
phone_idx = 0

# For custom audio, we assign score 0 to all phones (assuming correct pronunciation)
# The GOPT model will still evaluate actual pronunciation quality
for word in words:
    # Each word contributes some phones (approximate)
    word_upper = word.upper()
    # Use a simple estimate: ~3 phones per word on average
    # This doesn't need to be exact as the GOP extraction uses its own alignment
    scores[f"utt_001.{phone_idx}"] = 0
    phone_idx += 1

# Write scores.json
scores_path = "$KALDI_GOP_DIR/data/local/scores.json"
with open(scores_path, 'w') as f:
    json.dump(scores, f, indent=2)

print(f"Dummy scores generated with {len(scores)} entries at {scores_path}")
PYEOF

    print_success "Dummy scores generated"
}

# Step 5: Run Kaldi GOP extraction (fully automated)
run_kaldi_gop() {
    local dataset_name=$1
    
    print_info "Step 5: Running Kaldi GOP feature extraction (fully automated)"
    
    cd "$KALDI_GOP_DIR"
    
    # Source Kaldi environment
    . ./cmd.sh
    . ./path.sh
    
    local nj=1
    local model="$CHAIN_MODEL"
    local ivector_extractor="$IVECTOR_EXTRACTOR"
    local lang="$LANG_DIR"
    
    print_info "Using model: $model"
    print_info "Using ivector: $ivector_extractor"
    print_info "Using lang: $lang"
    
    # Step 3: Create high-resolution MFCC features
    print_info "Creating MFCC features..."
    for part in train test; do
        steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
            --cmd "$cmd" data/$dataset_name/$part || exit 1
        steps/compute_cmvn_stats.sh data/$dataset_name/$part || exit 1
        utils/fix_data_dir.sh data/$dataset_name/$part
    done
    print_success "MFCC features created"
    
    # Step 4: Extract ivector
    print_info "Extracting i-vectors..."
    for part in train test; do
        steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj $nj \
            data/$dataset_name/$part $ivector_extractor data/$dataset_name/$part/ivectors || exit 1
    done
    print_success "i-vectors extracted"
    
    # Step 5: Compute Log-likelihoods
    print_info "Computing log-likelihoods..."
    for part in train test; do
        steps/nnet3/compute_output.sh --cmd "$cmd" --nj $nj \
            --online-ivector-dir data/$dataset_name/$part/ivectors \
            data/$dataset_name/$part $model exp/probs_${dataset_name}_$part
    done
    print_success "Log-likelihoods computed"
    
    # Step 6: Prepare lang - always clean and rebuild for custom audio
    print_info "Preparing lang (clean rebuild)..."
    rm -rf data/lang_nosp data/local/dict_nosp data/local/lang_tmp_nosp
    local/prepare_dict.sh data/local/lexicon.txt data/local/dict_nosp || {
        print_error "prepare_dict.sh failed"
        return 1
    }
    
    utils/prepare_lang.sh --phone-symbol-table $lang/phones.txt \
        data/local/dict_nosp "<UNK>" data/local/lang_tmp_nosp data/lang_nosp || {
        print_error "prepare_lang.sh failed"
        return 1
    }
    
    if [ ! -f "data/lang_nosp/phones.txt" ]; then
        print_error "phones.txt not created after prepare_lang.sh"
        return 1
    fi
    print_success "Lang prepared with $(wc -l < data/lang_nosp/phones.txt) phones"
    
    # Now that lang_nosp/phones.txt exists, generate text-phone file
    # This needs to happen after prepare_lang.sh creates the phone table
    local transcript_text=$(head -1 data/$dataset_name/test/text | cut -d' ' -f2-)
    generate_text_phone "$dataset_name" "$transcript_text"
    
    # Step 7: Split data and make phone-level transcripts
    print_info "Preparing phone-level transcripts..."
    for part in train test; do
        utils/split_data.sh data/$dataset_name/$part $nj
        for i in $(seq 1 $nj); do
            utils/sym2int.pl -f 2- data/lang_nosp/words.txt \
                data/$dataset_name/$part/split${nj}/$i/text \
                > data/$dataset_name/$part/split${nj}/$i/text.int
        done
        
        utils/sym2int.pl -f 2- data/lang_nosp/phones.txt \
            data/local/text-phone > data/local/text-phone.int
    done
    print_success "Phone-level transcripts prepared"
    
    # Step 8: Make align graphs
    print_info "Making alignment graphs..."
    for part in train test; do
        mkdir -p exp/ali_${dataset_name}_$part/log
        $cmd JOB=1:$nj exp/ali_${dataset_name}_$part/log/mk_align_graph.JOB.log \
            compile-train-graphs-without-lexicon \
                --read-disambig-syms=data/lang_nosp/phones/disambig.int \
                $model/tree $model/final.mdl \
                "ark,t:data/$dataset_name/$part/split${nj}/JOB/text.int" \
                "ark,t:data/local/text-phone.int" \
                "ark:|gzip -c > exp/ali_${dataset_name}_$part/fsts.JOB.gz" || exit 1
        echo $nj > exp/ali_${dataset_name}_$part/num_jobs
    done
    print_success "Alignment graphs created"
    
    # Step 9: Align
    print_info "Running alignment..."
    for part in train test; do
        steps/align_mapped.sh --cmd "$cmd" --nj $nj \
            --graphs exp/ali_${dataset_name}_$part \
            data/$dataset_name/$part exp/probs_${dataset_name}_$part \
            $lang $model exp/ali_${dataset_name}_$part
    done
    print_success "Alignment completed"
    
    # Step 10: Make phone-to-pure-phone map (if not already done)
    if [ ! -f "data/lang_nosp/phone-to-pure-phone.int" ]; then
        print_info "Creating phone-to-pure-phone map..."
        local/remove_phone_markers.pl $lang/phones.txt \
            data/lang_nosp/phones-pure.txt data/lang_nosp/phone-to-pure-phone.int
        print_success "Phone map created"
    fi
    
    # Step 11: Convert transition-id to phone-id
    print_info "Converting transition-id to phone-id..."
    for part in train test; do
        mkdir -p exp/ali_${dataset_name}_$part/log
        $cmd JOB=1:$nj exp/ali_${dataset_name}_$part/log/ali_to_phones.JOB.log \
            ali-to-phones --per-frame=true $model/final.mdl \
                "ark,t:gunzip -c exp/ali_${dataset_name}_$part/ali.JOB.gz|" \
                "ark,t:|gzip -c >exp/ali_${dataset_name}_$part/ali-phone.JOB.gz" || exit 1
    done
    print_success "Phone conversion completed"
    
    # Step 12: Compute GOP
    print_info "Computing GOP features..."
    for part in train test; do
        mkdir -p exp/gop_${dataset_name}_$part/log
        $cmd JOB=1:$nj exp/gop_${dataset_name}_$part/log/compute_gop.JOB.log \
            compute-gop --phone-map=data/lang_nosp/phone-to-pure-phone.int \
                --skip-phones-string=0:1:2 \
                $model/final.mdl \
                "ark,t:gunzip -c exp/ali_${dataset_name}_$part/ali.JOB.gz|" \
                "ark,t:gunzip -c exp/ali_${dataset_name}_$part/ali-phone.JOB.gz|" \
                "ark:exp/probs_${dataset_name}_$part/output.JOB.ark" \
                "ark,scp:exp/gop_${dataset_name}_$part/gop.JOB.ark,exp/gop_${dataset_name}_$part/gop.JOB.scp" \
                "ark,scp:exp/gop_${dataset_name}_$part/feat.JOB.ark,exp/gop_${dataset_name}_$part/feat.JOB.scp" || exit 1
        cat exp/gop_${dataset_name}_$part/feat.*.scp > exp/gop_${dataset_name}_$part/feat.scp
        cat exp/gop_${dataset_name}_$part/gop.*.scp > exp/gop_${dataset_name}_$part/gop.scp
    done
    print_success "GOP features computed"
    
    # Store dataset name for later use
    echo "$dataset_name" > /tmp/current_dataset_name
}

# Step 6: Extract GOP features to CSV (custom, no human scores needed)
extract_gop_features() {
    local dataset_name=$1
    
    print_info "Step 6: Extracting GOP features to CSV (custom extractor)"
    
    cd "$KALDI_GOP_DIR"
    
    python3 - <<PYEOF
import sys, os
import numpy as np
import kaldi_io

dataset_name = "$dataset_name"

# Phone symbol table
phone_sym2int = {}
phone_int2sym = {}
phone_table_path = 'data/lang_nosp/phones-pure.txt'
if os.path.exists(phone_table_path):
    with open(phone_table_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                phone_symbol = parts[0]
                phone_id = int(parts[1])
                phone_sym2int[phone_symbol] = phone_id
                phone_int2sym[phone_id] = phone_symbol
    print(f"Loaded {len(phone_sym2int)} phone symbols")

os.makedirs('gopt_feats', exist_ok=True)

for set_name in ['train', 'test']:
    feat_scp = f'exp/gop_{dataset_name}_{set_name}/feat.scp'
    if not os.path.exists(feat_scp):
        print(f"Warning: {feat_scp} not found, skipping {set_name} set")
        continue
    
    keys = []
    features = []
    labels = []
    cnt = 0
    
    for key, feat in kaldi_io.read_vec_flt_scp(feat_scp):
        cnt += 1
        ph = int(feat[0])
        if ph in phone_int2sym:
            ph_sym = phone_int2sym[ph]
        else:
            ph_sym = f"PH{ph}"
        
        keys.append(key)
        features.append(feat)
        labels.append([ph_sym, 0])  # Dummy score of 0
    
    print(f'Processing {set_name} set: loaded {cnt} samples')
    
    if set_name == 'test':
        np.savetxt('gopt_feats/te_feats.csv', features, delimiter=',')
        np.savetxt('gopt_feats/te_keys_phn.csv', keys, delimiter=',', fmt='%s')
        np.savetxt('gopt_feats/te_labels_phn.csv', labels, delimiter=',', fmt='%s')
    elif set_name == 'train':
        np.savetxt('gopt_feats/tr_feats.csv', features, delimiter=',')
        np.savetxt('gopt_feats/tr_keys_phn.csv', keys, delimiter=',', fmt='%s')
        np.savetxt('gopt_feats/tr_labels_phn.csv', labels, delimiter=',', fmt='%s')

print("GOP features extracted to gopt_feats/")
PYEOF

    # Copy to GOPT data directory
    local target_dir="$GOPT_ROOT/data/raw_kaldi_gop/$dataset_name"
    mkdir -p "$target_dir"
    cp -r gopt_feats/* "$target_dir/"
    
    print_success "Features copied to: $target_dir"
}

# Step 7: Convert to sequence format
convert_to_sequence() {
    local dataset_name=$1
    
    print_info "Step 7: Converting GOP features to sequence format"
    
    cd "$GOPT_ROOT"
    
    python3 - <<PYEOF
import sys
import numpy as np
import os

def load_feat(path):
    return np.loadtxt(path, delimiter=',')

def load_keys(path):
    return np.loadtxt(path, delimiter=',', dtype=str)

def load_label(path):
    return np.loadtxt(path, delimiter=',', dtype=str)

def gen_phn_dict(label):
    phn_dict = {}
    phn_idx = 0
    for i in range(label.shape[0]):
        if label[i] not in phn_dict:
            phn_dict[label[i]] = phn_idx
            phn_idx += 1
    return phn_dict

def process_feat_seq(feat, keys, labels, phn_dict):
    key_set = []
    for i in range(keys.shape[0]):
        cur_key = keys[i].split('.')[0]
        key_set.append(cur_key)
    
    feat_dim = feat.shape[1] - 1
    utt_cnt = len(list(set(key_set)))
    
    print(f'Processing {utt_cnt} utterances, feature dim: {feat_dim}')
    
    seq_feat = np.zeros([utt_cnt, 50, feat_dim])
    seq_label = np.zeros([utt_cnt, 50, 2]) - 1
    
    prev_utt_id = keys[0].split('.')[0]
    row = 0
    
    for i in range(feat.shape[0]):
        cur_utt_id, cur_tok_id = keys[i].split('.')[0], int(keys[i].split('.')[1])
        if cur_utt_id != prev_utt_id:
            row += 1
            prev_utt_id = cur_utt_id
        
        seq_feat[row, cur_tok_id, :] = feat[i, 1:]
        seq_label[row, cur_tok_id, 0] = phn_dict[labels[i]]
    
    return seq_feat, seq_label

dataset = "$dataset_name"
base_path = "data/raw_kaldi_gop/" + dataset

print(f"Loading training data from {base_path}")
tr_feat = load_feat(base_path + '/tr_feats.csv')
tr_keys = load_keys(base_path + '/tr_keys_phn.csv')
tr_label = load_label(base_path + '/tr_labels_phn.csv')

phn_dict = gen_phn_dict(tr_label[:, 0])
tr_feat, tr_label = process_feat_seq(tr_feat, tr_keys, tr_label[:, 0], phn_dict)

output_dir = "data/seq_data_" + dataset
os.makedirs(output_dir, exist_ok=True)

np.save(output_dir + '/tr_feat.npy', tr_feat)
np.save(output_dir + '/tr_label_phn.npy', tr_label)

print(f"Loading test data from {base_path}")
te_feat = load_feat(base_path + '/te_feats.csv')
te_keys = load_keys(base_path + '/te_keys_phn.csv')
te_label = load_label(base_path + '/te_labels_phn.csv')

te_feat, te_label = process_feat_seq(te_feat, te_keys, te_label[:, 0], phn_dict)

np.save(output_dir + '/te_feat.npy', te_feat)
np.save(output_dir + '/te_label_phn.npy', te_label)

print(f"Sequence data saved to: {output_dir}")
print(f"Training shape: {tr_feat.shape}")
print(f"Test shape: {te_feat.shape}")
print(f"Feature dimension: {tr_feat.shape[2]} (should be 84 = 42 LPP + 42 LPR)")
PYEOF

    print_success "Sequence data generated"
}

# Step 8: Run GOPT inference
run_gopt_inference() {
    local dataset_name=$1
    local output_file=$2
    
    print_info "Step 8: Running GOPT inference"
    
    cd "$GOPT_ROOT"
    
    python3 - <<PYEOF
import sys
sys.path.append('src')
import torch
import numpy as np
from models import GOPT
import json

# Load model
print("Loading GOPT model...")
model = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
model = torch.nn.DataParallel(model)
checkpoint = torch.load('pretrained_models/gopt_librispeech/best_audio_model.pth', map_location='cpu')
model.load_state_dict(checkpoint, strict=True)
model.eval()

# Load features
dataset = "$dataset_name"
feat_path = f"data/seq_data_{dataset}/te_feat.npy"
phn_path = f"data/seq_data_{dataset}/te_label_phn.npy"

print(f"Loading features from {feat_path}")
feat = np.load(feat_path)
phn = np.load(phn_path)

print(f"Feature shape: {feat.shape}")
print(f"Phone shape: {phn.shape}")

# Normalize - only normalize VALID (non-zero) positions, keep padding as 0
# This matches the training normalization in inference_api.py._normalize_features
norm_mean, norm_std = 3.203, 4.045
feat_norm = np.zeros_like(feat)
for i in range(feat.shape[0]):
    for j in range(feat.shape[1]):
        if feat[i, j, 0] != 0:
            feat_norm[i, j, :] = (feat[i, j, :] - norm_mean) / norm_std
        else:
            break  # Stop at first zero (padding)

# Inference
print("Running inference...")
with torch.no_grad():
    feat_tensor = torch.FloatTensor(feat_norm)
    phn_tensor = torch.FloatTensor(phn[:, :, 0])
    
    u1, u2, u3, u4, u5, p, w1, w2, w3 = model(feat_tensor, phn_tensor)

# Extract results
results = {
    'utterance_scores': {
        'accuracy': float(u1.squeeze().numpy()),
        'completeness': float(u2.squeeze().numpy()),
        'fluency': float(u3.squeeze().numpy()),
        'prosodic': float(u4.squeeze().numpy()),
        'total': float(u5.squeeze().numpy())
    },
    'phone_scores': p.squeeze().numpy().tolist(),
    'word_scores': {
        'accuracy': w1.squeeze().numpy().tolist(),
        'stress': w2.squeeze().numpy().tolist(),
        'total': w3.squeeze().numpy().tolist()
    }
}

# Save results
output_file = "$output_file"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_file}")
print("\n" + "="*60)
print("PRONUNCIATION ASSESSMENT RESULTS")
print("="*60)
print("\nUtterance-Level Scores (0-2 scale):")
for aspect, score in results['utterance_scores'].items():
    print(f"  {aspect:15s}: {score:.3f}")

print("\nPhone-Level Scores (first 10 phones):")
for i, score in enumerate(results['phone_scores'][:10]):
    if score > 0:
        print(f"  Phone {i+1:2d}: {score:.3f}")

print("\n" + "="*60)
PYEOF

    print_success "Inference completed!"
}

# Main function
main() {
    if [ "$#" -lt 2 ]; then
        echo "Usage: $0 <audio_file> <transcript> [dataset_name]"
        echo ""
        echo "Example:"
        echo "  $0 /workspace/audio_input/test.wav \"HELLO WORLD\" my_test"
        echo ""
        echo "Arguments:"
        echo "  audio_file    - Path to WAV audio file (16kHz mono WAV)"
        echo "  transcript    - Text transcript (will be converted to uppercase)"
        echo "  dataset_name  - Optional custom dataset name (default: custom_audio)"
        exit 1
    fi
    
    local audio_file=$1
    local transcript=$2
    local dataset_name=${3:-custom_audio}
    local output_file="/workspace/audio_output/${dataset_name}_results.json"
    
    echo ""
    echo "========================================================================"
    echo "           GOPT Pronunciation Assessment Pipeline (FULLY AUTOMATED)"
    echo "========================================================================"
    echo ""
    echo "Audio File:    $audio_file"
    echo "Transcript:    $transcript"
    echo "Dataset Name:  $dataset_name"
    echo "Output File:   $output_file"
    echo ""
    echo "========================================================================"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Validate audio
    validate_audio "$audio_file"
    
    # Run pipeline steps
    prepare_kaldi_data "$audio_file" "$transcript" "$dataset_name"
    generate_lexicon "$dataset_name" "$transcript"
    generate_dummy_scores "$dataset_name" "$transcript"
    # Note: generate_text_phone is called INSIDE run_kaldi_gop after lang prep
    # because it needs data/lang_nosp/phones.txt to exist
    run_kaldi_gop "$dataset_name"
    extract_gop_features "$dataset_name"
    convert_to_sequence "$dataset_name"
    run_gopt_inference "$dataset_name" "$output_file"
    
    print_success "========================================================================"
    print_success "Pipeline completed successfully!"
    print_success "Results saved to: $output_file"
    print_success "========================================================================"
}

# Run main function
main "$@"