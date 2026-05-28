FROM kaldiasr/kaldi:latest

# Metadata
LABEL maintainer="GOPT Project"
LABEL description="Docker image for GOPT pronunciation assessment with full Kaldi pipeline"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV KALDI_ROOT=/opt/kaldi
ENV PATH=$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PATH
ENV LD_LIBRARY_PATH=$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH

# Install build dependencies and other tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    git \
    wget \
    curl \
    vim \
    dos2unix \
    nano \
    sox \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Download and compile Python 3.8.18 from source
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/3.8.18/Python-3.8.18.tgz && \
    tar xzf Python-3.8.18.tgz && \
    cd Python-3.8.18 && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j$(nproc) && \
    make altinstall && \
    cd /tmp && \
    rm -rf Python-3.8.18 Python-3.8.18.tgz

# Set python3.8 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.8 1 && \
    update-alternatives --set python3 /usr/local/bin/python3.8 && \
    ln -sf /usr/local/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.8 /usr/bin/pip && \
    ln -sf /usr/local/bin/pip3.8 /usr/bin/pip3

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create working directory
WORKDIR /workspace/gopt

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies for GOP extraction scripts
RUN pip3 install --no-cache-dir imbalanced-learn scikit-learn pandas

# Install Whisper for ASR (automatic speech recognition)
# This will download the base model (~150MB) on first use
RUN pip3 install --no-cache-dir openai-whisper numba

# Copy GOPT project files
COPY . .

# Ensure models.py is available in all import locations
RUN cp models.py src/models.py 2>/dev/null || true && \
    cp models.py pretrained_models/models.py 2>/dev/null || true

# Create necessary directories
RUN mkdir -p \
    /workspace/gopt/data/raw_kaldi_gop/custom_audio \
    /workspace/gopt/data/seq_data_custom_audio \
    /workspace/gopt/data/temp_custom_audio \
    /workspace/gopt/exp/custom_audio \
    /workspace/models/librispeech \
    /workspace/audio_input \
    /workspace/audio_output

# Copy extraction scripts to Kaldi
RUN if [ -d "$KALDI_ROOT/egs/gop_speechocean762/s5" ]; then \
        cp src/extract_kaldi_gop/extract_gop_feats.py $KALDI_ROOT/egs/gop_speechocean762/s5/local/ && \
        cp src/extract_kaldi_gop/extract_gop_feats_word.py $KALDI_ROOT/egs/gop_speechocean762/s5/local/ && \
        cp src/extract_kaldi_gop/utils.py $KALDI_ROOT/egs/gop_speechocean762/s5/local/ && \
        echo "GOPT extraction scripts copied successfully"; \
    fi

# Note: GOP computation is integrated into the Kaldi recipe workflow (run.sh)
# The gop_speechocean762 recipe includes all necessary GOP extraction functionality

# Create helper scripts directory
RUN mkdir -p /workspace/scripts

# Copy Docker helper scripts (will be created separately)
COPY docker/*.sh /workspace/scripts/
RUN dos2unix /workspace/scripts/*.sh && \
    chmod +x /workspace/scripts/*.sh

# Copy and setup the entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN dos2unix /entrypoint.sh && chmod +x /entrypoint.sh

# Copy the fixed audio processing script
COPY process_custom_audio_fixed.sh /workspace/gopt/process_custom_audio.sh
RUN dos2unix /workspace/gopt/process_custom_audio.sh && \
    chmod +x /workspace/gopt/process_custom_audio.sh

# Set up volumes for:
# - Audio input/output
# - Librispeech models
# - Results and logs
VOLUME ["/workspace/audio_input", "/workspace/audio_output", "/workspace/models", "/workspace/gopt/exp"]

# Expose port for potential future web interface
EXPOSE 8000

# Set entrypoint for auto-setup
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; import numpy; print('OK')" || exit 1

