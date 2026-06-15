#!/usr/bin/env python
"""
Debug script to check file structure in QQ Music directory.
"""

from pathlib import Path
import os

def check_directory(music_dir):
    music_dir = Path(music_dir)
    
    print(f"Checking directory: {music_dir}")
    print(f"Directory exists: {music_dir.exists()}")
    print("=" * 80)
    
    # List first 10 LRC files
    lrc_files = list(music_dir.glob('*.lrc'))
    print(f"\nTotal LRC files: {len(lrc_files)}")
    print(f"First 10 LRC files:")
    for f in lrc_files[:10]:
        print(f"  {f.name}")
    
    # List first 10 audio files
    audio_files = []
    for ext in ['*.ogg', '*.mgg', '*.mp3', '*.flac', '*.wav']:
        audio_files.extend(music_dir.glob(ext))
    
    print(f"\nTotal audio files: {len(audio_files)}")
    print(f"First 10 audio files:")
    for f in audio_files[:10]:
        print(f"  {f.name}")
    
    # Try matching first LRC with audio
    if lrc_files:
        print("\n" + "=" * 80)
        print("Testing file matching for first LRC file:")
        lrc = lrc_files[0]
        print(f"\nLRC file: {lrc.name}")
        print(f"LRC stem (no ext): {lrc.stem}")
        
        # Check if audio with exact name exists
        for ext in ['.ogg', '.mgg', '.mp3', '.flac', '.wav']:
            audio_candidate = lrc.with_suffix(ext)
            exists = audio_candidate.exists()
            print(f"  Checking {audio_candidate.name}: {'EXISTS' if exists else 'NOT FOUND'}")
        
        # Check if removing _L works
        if lrc.stem.endswith('_L'):
            clean_name = lrc.stem[:-2]
            print(f"\nTrying without _L suffix: {clean_name}")
            for ext in ['.ogg', '.mgg', '.mp3', '.flac', '.wav']:
                audio_candidate = music_dir / f"{clean_name}{ext}"
                exists = audio_candidate.exists()
                print(f"  Checking {audio_candidate.name}: {'EXISTS' if exists else 'NOT FOUND'}")
        
        # List files with similar prefix
        prefix = lrc.stem[:-2] if lrc.stem.endswith('_L') else lrc.stem
        print(f"\nFiles with prefix '{prefix}':")
        for f in music_dir.iterdir():
            if f.name.startswith(prefix) and f.suffix in ['.ogg', '.mgg', '.mp3', '.flac', '.wav', '.lrc']:
                print(f"  {f.name}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        check_directory(sys.argv[1])
    else:
        print("Usage: python debug_qqmusic.py <music_dir>")
