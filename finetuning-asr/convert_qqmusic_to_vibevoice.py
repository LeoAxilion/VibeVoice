#!/usr/bin/env python
"""
Convert QQ Music LRC + OGG/MGG files to VibeVoice ASR fine-tuning format.

This script processes QQ Music downloads (LRC lyrics + audio files) and converts them
into the JSON format required by VibeVoice ASR's lora_finetune.py.

Features:
- Parses LRC lyric files with timestamp extraction
- Handles OGG/MGG audio format conversion (MGG → OGG if needed)
- Generates VibeVoice-compatible training data
- Supports batch processing of multiple songs

Usage:
    python convert_qqmusic_to_vibevoice.py \
        --music_dir /path/to/qqmusic_folder \
        --output_dir ./qqmusic_vibevoice_format \
        --audio_format ogg

Requirements:
    pip install pydub tqdm
    # Install ffmpeg for audio conversion
    # Ubuntu/Debian: sudo apt install ffmpeg
    # macOS: brew install ffmpeg
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


def parse_lrc(lrc_path: str) -> List[Dict]:
    """
    Parse LRC lyric file and extract timestamps with text.
    
    LRC format:
    [mm:ss.xx]Lyric text here
    [mm:ss.xx][mm:ss.xx]Multiple timestamps for same line (repeated)
    
    Returns:
        List of dicts with keys: 'start', 'text'
    """
    segments = []
    
    # Try multiple encodings: UTF-8, GBK, GB2312, GB18030
    content = None
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(lrc_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if content is None:
        print(f"  Warning: Could not decode {lrc_path} with any supported encoding")
        return segments
    
    # Remove metadata tags like [ti:Title], [ar:Artist], etc.
    content = re.sub(r'\[(ti|ar|al|by|offset|length):[^\]]*\]', '', content)
    
    # Find all timestamp + text patterns
    # Pattern: [mm:ss.xx]text or [mm:ss.xx][mm:ss.xx]text
    pattern = r'\[(\d+):(\d+\.\d+)\](.*?)$'
    
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Find all timestamps in this line
        timestamps = re.findall(r'\[(\d+):(\d+\.\d+)\]', line)
        
        # Get text after all timestamps
        text_match = re.search(r'(?:\[\d+:\d+\.\d+\])*(.*)', line)
        if not text_match:
            continue
        
        text = text_match.group(1).strip()
        if not text or text.startswith('['):
            continue
        
        # Create segment for each timestamp
        for minutes, seconds in timestamps:
            start_time = int(minutes) * 60 + float(seconds)
            segments.append({
                'start': round(start_time, 2),
                'text': text,
            })
    
    # Sort by start time
    segments.sort(key=lambda x: x['start'])
    
    return segments


def estimate_segment_end_times(segments: List[Dict], audio_duration: float) -> List[Dict]:
    """
    Estimate end times for lyric segments.
    
    Since LRC only provides start times, we estimate end times by:
    1. Using the gap between consecutive segments
    2. For the last segment, using remaining audio time or a fixed duration
    """
    if not segments:
        return segments
    
    enhanced_segments = []
    
    for i, seg in enumerate(segments):
        enhanced_seg = seg.copy()
        
        if i < len(segments) - 1:
            # End time is the start of next segment minus a small gap
            next_start = segments[i + 1]['start']
            current_start = seg['start']
            gap = next_start - current_start
            
            # The actual singing duration is usually 60-80% of the gap
            estimated_duration = gap * 0.7
            enhanced_seg['end'] = round(current_start + estimated_duration, 2)
        else:
            # Last segment
            enhanced_seg['end'] = min(
                round(seg['start'] + 5.0, 2),  # Default 5 seconds
                round(audio_duration, 2)
            )
        
        enhanced_segments.append(enhanced_seg)
    
    return enhanced_segments


def get_audio_duration(audio_path: str) -> Optional[float]:
    """
    Get audio file duration in seconds using ffprobe.
    
    Returns:
        Duration in seconds, or None if extraction fails
    """
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
            ],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        pass
    
    return None


def convert_mgg_to_ogg(mgg_path: str, ogg_path: str) -> bool:
    """
    Convert MGG format to OGG using ffmpeg.
    
    Note: MGG is a Tencent proprietary format. If ffmpeg cannot decode it directly,
    you may need to use QQ Music's official player to export or use a conversion tool.
    
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Try direct conversion
        result = subprocess.run(
            [
                'ffmpeg', '-y', '-i', mgg_path, '-c:a', 'libvorbis',
                '-q:a', '6', ogg_path
            ],
            capture_output=True, text=True, timeout=120
        )
        
        if result.returncode == 0 and os.path.exists(ogg_path):
            return True
        
        return False
        
    except Exception as e:
        return False


def try_get_duration_with_conversion(audio_path: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Try to get audio duration, converting MGG to OGG if needed.
    
    Returns:
        Tuple of (duration_in_seconds, converted_audio_path)
        If conversion was needed, converted_audio_path will be the new OGG path.
    """
    audio_path = Path(audio_path)
    
    # First try direct duration
    duration = get_audio_duration(str(audio_path))
    if duration is not None:
        return duration, None
    
    # If failed and it's MGG, try converting
    if audio_path.suffix.lower() == '.mgg':
        ogg_path = str(audio_path.with_suffix('.ogg'))
        
        # Check if OGG already exists from previous conversion
        if os.path.exists(ogg_path):
            duration = get_audio_duration(ogg_path)
            if duration is not None:
                return duration, ogg_path
        
        # Try converting
        if convert_mgg_to_ogg(str(audio_path), ogg_path):
            duration = get_audio_duration(ogg_path)
            if duration is not None:
                return duration, ogg_path
    
    return None, None


def find_audio_file(lrc_path: Path, directory: Path) -> Optional[Path]:
    """
    Find audio file matching the LRC file.
    QQ Music files have exact same name with different extension.
    """
    # Try exact match first (replace .lrc with audio extension)
    for ext in ['.ogg', '.mgg', '.mp3', '.flac', '.wav']:
        audio_path = lrc_path.with_suffix(ext)
        if audio_path.exists():
            return audio_path
    
    return None



def find_lrc_file(base_name: str, directory: Path) -> Optional[Path]:
    """
    Find LRC file matching the base name.
    """
    lrc_path = directory / f"{base_name}.lrc"
    if lrc_path.exists():
        return lrc_path
    
    # Try with different naming patterns
    for pattern in [f"{base_name}*.lrc", f"*{base_name}*.lrc"]:
        matches = list(directory.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def process_single_song(
    lrc_path: Path,
    audio_path: Path,
    output_dir: Path,
    song_index: int,
) -> Optional[Dict]:
    """
    Process a single song: parse LRC, get audio info, create VibeVoice format.
    
    Returns:
        Dict with song metadata and segment info, or None if processing failed
    """
    base_name = lrc_path.stem
    
    # Get audio duration, converting MGG if needed
    audio_duration, converted_audio = try_get_duration_with_conversion(str(audio_path))
    if audio_duration is None:
        return None
    
    # If MGG was converted, use the converted OGG
    if converted_audio:
        audio_path = Path(converted_audio)
    
    # Parse LRC
    segments = parse_lrc(str(lrc_path))
    if not segments:
        return None
    
    # Estimate end times
    segments = estimate_segment_end_times(segments, audio_duration)
    
    # Create output audio path (relative to output directory)
    output_audio_name = f"{song_index:04d}_{audio_path.name}"
    output_audio_path = output_dir / output_audio_name
    
    # Copy or convert audio file
    if audio_path.suffix.lower() == '.mgg':
        ogg_path = output_dir / f"{song_index:04d}_{base_name}.ogg"
        if not convert_mgg_to_ogg(str(audio_path), str(ogg_path)):
            return None
        output_audio_name = f"{song_index:04d}_{base_name}.ogg"
    else:
        # Copy OGG/MP3/etc
        import shutil
        shutil.copy2(audio_path, output_audio_path)
    
    # Create VibeVoice format data
    vibevoice_data = {
        'audio_duration': round(audio_duration, 2),
        'audio_path': output_audio_name,
        'segments': [
            {
                'speaker': 0,  # Music typically has one "speaker" (singer)
                'text': seg['text'],
                'start': seg['start'],
                'end': seg['end'],
            }
            for seg in segments
        ],
        'customized_context': [],  # Can add song title, artist name, etc.
    }
    
    # Save JSON
    output_json = output_dir / f"{song_index:04d}_{base_name}.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(vibevoice_data, f, ensure_ascii=False, indent=2)
    
    return {
        'song_name': base_name,
        'duration': audio_duration,
        'segments_count': len(segments),
        'json_path': str(output_json),
    }


def convert_qqmusic_to_vibevoice(
    music_dir: str,
    output_dir: str,
    audio_format: str = 'ogg',
    max_songs: int = None,
):
    """
    Main conversion function.
    
    Args:
        music_dir: Directory containing QQ Music files (LRC + audio)
        output_dir: Output directory for VibeVoice format
        audio_format: Preferred audio format (ogg, mgg, etc.)
        max_songs: Maximum number of songs to process (for testing)
    """
    music_dir = Path(music_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not music_dir.exists():
        raise FileNotFoundError(f"Music directory not found: {music_dir}")
    
    # Find all LRC files
    lrc_files = sorted(music_dir.rglob('*.lrc'))
    
    if not lrc_files:
        print(f"No LRC files found in {music_dir}")
        return
    
    if max_songs:
        lrc_files = lrc_files[:max_songs]
    
    print(f"Found {len(lrc_files)} LRC files to process")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    processed_songs = []
    skipped_songs = []
    
    for idx, lrc_path in enumerate(tqdm(lrc_files, desc="Processing songs")):
        try:
            base_name = lrc_path.stem
            
            # Find corresponding audio file
            audio_path = find_audio_file(lrc_path, lrc_path.parent)
            if audio_path is None:
                print(f"\nWarning: No audio file found for {base_name}")
                skipped_songs.append(str(lrc_path))
                continue
            
            # Process song
            result = process_single_song(
                lrc_path=lrc_path,
                audio_path=audio_path,
                output_dir=output_dir,
                song_index=len(processed_songs),
            )
            
            if result:
                processed_songs.append(result)
            else:
                skipped_songs.append(str(lrc_path))
        
        except Exception as e:
            print(f"\nError processing {lrc_path}: {e}")
            skipped_songs.append(str(lrc_path))
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"  Processed: {len(processed_songs)} songs")
    print(f"  Skipped: {len(skipped_songs)} songs")
    print(f"  Total segments: {sum(s['segments_count'] for s in processed_songs)}")
    print(f"  Total duration: {sum(s['duration'] for s in processed_songs) / 3600:.2f} hours")
    print(f"  Output directory: {output_dir}")
    
    if skipped_songs:
        print(f"\nSkipped files:")
        for path in skipped_songs[:10]:
            print(f"  - {path}")
        if len(skipped_songs) > 10:
            print(f"  ... and {len(skipped_songs) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description='Convert QQ Music (LRC + OGG/MGG) to VibeVoice ASR fine-tuning format'
    )
    
    parser.add_argument(
        '--music_dir',
        type=str,
        required=True,
        help='Directory containing QQ Music files (LRC + audio)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./qqmusic_vibevoice_format',
        help='Output directory for converted files'
    )
    parser.add_argument(
        '--audio_format',
        type=str,
        default='ogg',
        choices=['ogg', 'mgg', 'mp3', 'flac'],
        help='Preferred audio format'
    )
    parser.add_argument(
        '--max_songs',
        type=int,
        default=None,
        help='Maximum number of songs to process (for testing)'
    )
    
    args = parser.parse_args()
    
    convert_qqmusic_to_vibevoice(
        music_dir=args.music_dir,
        output_dir=args.output_dir,
        audio_format=args.audio_format,
        max_songs=args.max_songs,
    )


if __name__ == '__main__':
    main()
