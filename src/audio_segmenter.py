#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Segmentation Module

This module splits audio files into segments based on text segment proportions.
Uses ffmpeg for precise audio manipulation.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class AudioSegment:
    """Represents an audio segment with metadata."""
    path: Path
    start_time: float
    duration: float
    segment_index: int
    
    def __repr__(self):
        return f"AudioSegment(seg{self.segment_index}, {self.duration:.2f}s)"


class AudioSegmenter:
    """
    Split audio files based on text segment proportions.
    
    Uses ffmpeg to split audio while maintaining quality.
    Adds small overlaps at boundaries to avoid cutting off speech.
    """
    
    def __init__(self, docker_container: str = "gopt-pipeline", overlap_seconds: float = 0.1):
        """
        Initialize the audio segmenter.
        
        Args:
            docker_container: Name of Docker container with ffmpeg
            overlap_seconds: Overlap duration at segment boundaries (default: 0.1s)
        """
        self.docker_container = docker_container
        self.overlap_seconds = overlap_seconds
    
    def get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """
        Get audio file duration using ffprobe.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds, or None if failed
        """
        try:
            # Path in container
            container_path = f"/workspace/audio_input/{audio_path.name}"
            
            result = subprocess.run(
                ["docker", "exec", self.docker_container,
                 "ffprobe", "-v", "error",
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1",
                 container_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                return duration
            else:
                print(f"Warning: Could not get audio duration: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return None
    
    def split_audio_by_text_ratio(
        self,
        audio_path: Path,
        text_segments: List,
        output_dir: Optional[Path] = None
    ) -> List[AudioSegment]:
        """
        Split audio based on text segment word counts.
        
        Strategy:
        - Calculate word count ratio for each segment
        - Allocate audio time proportionally
        - Add small overlap at boundaries to avoid cutting speech
        
        Args:
            audio_path: Path to source audio file
            text_segments: List of TextSegment objects
            output_dir: Output directory (default: audio_input/)
            
        Returns:
            List of AudioSegment objects with split audio files
        """
        if output_dir is None:
            output_dir = audio_path.parent
        
        # Get total audio duration
        total_duration = self.get_audio_duration(audio_path)
        if total_duration is None:
            raise RuntimeError(f"Could not determine audio duration for {audio_path}")
        
        # Calculate total words
        total_words = sum(seg.word_count for seg in text_segments)
        
        # Calculate time allocation for each segment
        time_allocations = []
        cumulative_time = 0.0
        
        for seg in text_segments:
            # Proportional time based on word count
            segment_duration = (seg.word_count / total_words) * total_duration
            time_allocations.append({
                'start': max(0, cumulative_time - self.overlap_seconds),
                'duration': segment_duration + (2 * self.overlap_seconds if cumulative_time > 0 else self.overlap_seconds),
                'segment': seg
            })
            cumulative_time += segment_duration
        
        # Adjust last segment to include any remaining time
        if time_allocations:
            last = time_allocations[-1]
            last['duration'] = total_duration - last['start']
        
        # Split audio using ffmpeg
        audio_segments = []
        base_name = audio_path.stem
        
        for i, alloc in enumerate(time_allocations):
            segment_path = output_dir / f"{base_name}_seg{i}.wav"
            
            success = self._split_audio_segment(
                audio_path,
                segment_path,
                start_time=alloc['start'],
                duration=alloc['duration']
            )
            
            if success:
                audio_segments.append(AudioSegment(
                    path=segment_path,
                    start_time=alloc['start'],
                    duration=alloc['duration'],
                    segment_index=i
                ))
            else:
                raise RuntimeError(f"Failed to split audio segment {i}")
        
        return audio_segments
    
    def _split_audio_segment(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        duration: float
    ) -> bool:
        """
        Split a single audio segment using ffmpeg.
        
        Args:
            input_path: Source audio file
            output_path: Output audio file
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Paths in container
            container_input = f"/workspace/audio_input/{input_path.name}"
            container_output = f"/workspace/audio_input/{output_path.name}"
            
            # Use ffmpeg to extract segment
            cmd = [
                "docker", "exec", self.docker_container,
                "ffmpeg",
                "-i", container_input,
                "-ss", str(start_time),
                "-t", str(duration),
                "-ar", "16000",  # Maintain 16kHz sample rate
                "-ac", "1",       # Maintain mono
                "-acodec", "pcm_s16le",  # PCM format
                "-y",  # Overwrite output
                container_output
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Verify the output file was created
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    print(f"Segment created: {output_path.name} ({file_size} bytes, {start_time:.2f}s-{start_time+duration:.2f}s)")
                    return True
                else:
                    print(f"FFmpeg succeeded but file not found: {output_path}")
                    return False
            else:
                print(f"FFmpeg error: {result.stderr[-500:]}")
                return False
                
        except Exception as e:
            print(f"Error splitting audio segment: {e}")
            return False
    
    def cleanup_segments(self, segments: List[AudioSegment]):
        """
        Clean up temporary segment files.
        
        Args:
            segments: List of AudioSegment objects to clean up
        """
        for seg in segments:
            try:
                if seg.path.exists():
                    seg.path.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {seg.path}: {e}")


def main():
    """Demo usage of AudioSegmenter."""
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    from text_segmenter import TextSegmenter, TextSegment
    
    print("=" * 70)
    print("Audio Segmentation Demo")
    print("=" * 70)
    
    # Example: simulate text segments
    text_segments = [
        TextSegment(
            text="HI I AM GORDON MY ESSAY IS ABOUT EARLY MORNING CLASSES",
            words=["HI", "I", "AM", "GORDON", "MY", "ESSAY", "IS", "ABOUT", "EARLY", "MORNING", "CLASSES"],
            start_word_idx=0,
            end_word_idx=10,
            phoneme_count=38,
            word_count=11
        ),
        TextSegment(
            text="AND HOW THEY AFFECT STUDENTS PERFORMANCE AND HEALTH",
            words=["AND", "HOW", "THEY", "AFFECT", "STUDENTS", "PERFORMANCE", "AND", "HEALTH"],
            start_word_idx=11,
            end_word_idx=18,
            phoneme_count=38,
            word_count=8
        )
    ]
    
    print(f"\nText segments: {len(text_segments)}")
    for i, seg in enumerate(text_segments, 1):
        print(f"  Segment {i}: {seg.word_count} words, ~{seg.phoneme_count} phonemes")
    
    print("\nAudio segmentation would split the audio proportionally:")
    total_words = sum(seg.word_count for seg in text_segments)
    for i, seg in enumerate(text_segments, 1):
        ratio = seg.word_count / total_words
        print(f"  Segment {i}: {ratio*100:.1f}% of audio duration")
    
    print("\n" + "=" * 70)
    print("Note: Actual audio splitting requires audio file and Docker container")
    print("=" * 70)


if __name__ == '__main__':
    main()

