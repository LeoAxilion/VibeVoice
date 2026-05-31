#!/usr/bin/env python
"""
Convert AMI Meeting Corpus dataset to VibeVoice ASR fine-tuning format.

This script converts AMI RTTM (speaker diarization) and STM (transcription) files
into the JSON format expected by VibeVoice ASR's lora_finetune.py.

Usage:
    python convert_ami_to_vibevoice.py \
        --ami_dir /path/to/ami/corpus \
        --output_dir ./ami_vibevoice_format \
        --max_duration 600  # 10 minutes max per file

Requirements:
    pip install pyaml tqdm
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def parse_rttm(rttm_path):
    """
    Parse RTTM file for speaker diarization.
    
    RTTM format:
    SPEAKER <file-id> <channel-id> <start-time> <duration> <NA> <NA> <speaker-id> <NA> <NA>
    """
    segments = []
    
    with open(rttm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            parts = line.split()
            if len(parts) < 9 or parts[0] != 'SPEAKER':
                continue
            
            file_id = parts[1]
            start_time = float(parts[3])
            duration = float(parts[4])
            speaker_id = parts[7]
            
            segments.append({
                'file_id': file_id,
                'start': start_time,
                'end': start_time + duration,
                'speaker': speaker_id,
            })
    
    return segments


def parse_stm(stm_path):
    """
    Parse STM file for transcriptions.
    
    STM format:
    <file-id> <channel-id> <speaker-id> <start-time> <end-time> <label> <transcript>
    """
    segments = []
    
    with open(stm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            
            file_id = parts[0]
            speaker_id = parts[2]
            start_time = float(parts[3])
            end_time = float(parts[4])
            text = parts[5].strip()
            
            # Skip non-speech segments
            if text in ['<F0_ASIDE>', '</F0_ASIDE>', '<NOISE>', '<SPOKEN_WORD>', '']:
                continue
            
            segments.append({
                'file_id': file_id,
                'start': start_time,
                'end': end_time,
                'speaker': speaker_id,
                'text': text,
            })
    
    return segments


def merge_diarization_and_transcription(rttm_segments, stm_segments):
    """
    Merge speaker diarization (RTTM) with transcriptions (STM).
    
    This function aligns speaker IDs from RTTM with text from STM,
    then merges consecutive segments from the same speaker.
    """
    # Create a mapping from speaker names to integer IDs
    speaker_map = {}
    speaker_id_counter = 0
    
    # Combine all segments and sort by start time
    all_segments = []
    
    for stm_seg in stm_segments:
        all_segments.append({
            'start': stm_seg['start'],
            'end': stm_seg['end'],
            'speaker': stm_seg['speaker'],
            'text': stm_seg['text'],
        })
    
    # Sort by start time
    all_segments.sort(key=lambda x: x['start'])
    
    # Assign integer IDs to speakers
    merged_segments = []
    for seg in all_segments:
        speaker_name = seg['speaker']
        if speaker_name not in speaker_map:
            speaker_map[speaker_name] = speaker_id_counter
            speaker_id_counter += 1
        
        merged_segments.append({
            'speaker': speaker_map[speaker_name],
            'text': seg['text'],
            'start': round(seg['start'], 2),
            'end': round(seg['end'], 2),
        })
    
    # Optionally merge consecutive segments from the same speaker
    # (within a small gap threshold)
    final_segments = []
    merge_gap_threshold = 0.5  # seconds
    
    for seg in merged_segments:
        if (final_segments and 
            final_segments[-1]['speaker'] == seg['speaker'] and
            seg['start'] - final_segments[-1]['end'] < merge_gap_threshold):
            # Merge with previous segment
            final_segments[-1]['text'] += ' ' + seg['text']
            final_segments[-1]['end'] = seg['end']
        else:
            final_segments.append(seg)
    
    return final_segments


def get_audio_duration(audio_path):
    """
    Get audio file duration in seconds using sox or ffprobe.
    """
    try:
        import subprocess
        
        # Try ffprobe first
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    
    # Fallback: return None if duration cannot be determined
    return None


def convert_ami_to_vibevoice(
    ami_dir: str,
    output_dir: str,
    max_duration: float = None,
    max_files: int = None,
):
    """
    Convert AMI dataset to VibeVoice ASR fine-tuning format.
    
    Args:
        ami_dir: Path to AMI corpus directory
        output_dir: Output directory for converted files
        max_duration: Maximum audio duration in seconds (skip longer files)
        max_files: Maximum number of files to process (for testing)
    """
    ami_dir = Path(ami_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all STM files
    stm_files = sorted(ami_dir.rglob('*.stm'))
    
    if max_files:
        stm_files = stm_files[:max_files]
    
    print(f"Found {len(stm_files)} STM files to process")
    
    processed_count = 0
    skipped_count = 0
    
    for stm_path in tqdm(stm_files, desc="Converting AMI to VibeVoice format"):
        try:
            # Get base filename
            base_name = stm_path.stem
            
            # Find corresponding RTTM file
            rttm_path = stm_path.with_suffix('.rttm')
            if not rttm_path.exists():
                # Try searching in parent directory or adjacent folders
                rttm_candidates = list(ami_dir.rglob(f'{base_name}.rttm'))
                if rttm_candidates:
                    rttm_path = rttm_candidates[0]
                else:
                    print(f"Warning: No RTTM file found for {base_name}")
                    skipped_count += 1
                    continue
            
            # Find audio file
            audio_candidates = (
                list(ami_dir.rglob(f'{base_name}.wav')) +
                list(ami_dir.rglob(f'{base_name}.mp3'))
            )
            if not audio_candidates:
                print(f"Warning: No audio file found for {base_name}")
                skipped_count += 1
                continue
            audio_path = audio_candidates[0]
            
            # Parse STM (transcription)
            stm_segments = parse_stm(str(stm_path))
            if not stm_segments:
                skipped_count += 1
                continue
            
            # Calculate audio duration
            audio_duration = get_audio_duration(str(audio_path))
            if audio_duration is None:
                # Use last segment's end time as approximation
                audio_duration = max(seg['end'] for seg in stm_segments)
            
            # Skip if too long
            if max_duration and audio_duration > max_duration:
                skipped_count += 1
                continue
            
            # Parse RTTM (diarization) - optional, STM already has speaker info
            # rttm_segments = parse_rttm(str(rttm_path))
            
            # Merge and format segments
            formatted_segments = merge_diarization_and_transcription(
                [],  # RTTM not needed if STM has speaker info
                stm_segments
            )
            
            # Get relative audio path
            rel_audio_path = os.path.relpath(audio_path, output_dir)
            
            # Create VibeVoice format JSON
            vibevoice_data = {
                'audio_duration': round(audio_duration, 2),
                'audio_path': rel_audio_path,
                'segments': formatted_segments,
                'customized_context': [],  # Optional: add domain-specific terms
            }
            
            # Save JSON
            output_json = output_dir / f'{base_name}.json'
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(vibevoice_data, f, ensure_ascii=False, indent=2)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {stm_path}: {e}")
            skipped_count += 1
            continue
    
    print(f"\nConversion complete!")
    print(f"  Processed: {processed_count} files")
    print(f"  Skipped: {skipped_count} files")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert AMI Meeting Corpus to VibeVoice ASR fine-tuning format'
    )
    
    parser.add_argument(
        '--ami_dir',
        type=str,
        required=True,
        help='Path to AMI corpus directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./ami_vibevoice_format',
        help='Output directory for converted files'
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        default=None,
        help='Maximum audio duration in seconds (skip longer files)'
    )
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Maximum number of files to process (for testing)'
    )
    
    args = parser.parse_args()
    
    convert_ami_to_vibevoice(
        ami_dir=args.ami_dir,
        output_dir=args.output_dir,
        max_duration=args.max_duration,
        max_files=args.max_files,
    )


if __name__ == '__main__':
    main()
