import os
import re
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def parse_filename(filename):
    """Parse filename to extract ID, name, version info"""
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]

    # Extract ID (first 5 digits)
    match = re.match(r'^(\d{5})_(.+)', name_without_ext)
    if not match:
        return None

    file_id = match.group(1)
    rest = match.group(2)

    return {
        'id': file_id,
        'filename': filename,
        'name_rest': rest,
        'full_name': name_without_ext
    }

def extract_version_info(name_rest, full_name):
    """Extract version number or date from filename"""
    # Look for version patterns like v01, v02, v_00001, etc.
    version_patterns = [
        r'_v(\d+)',           # _v01, _v02
        r'_v_(\d+)',          # _v_00001
        r'v(\d+)',            # v01 (without underscore)
        r'_(\d{5})$',         # ending with 5 digits (like _00001)
        r'_(\d{4})$',         # ending with 4 digits (like _0001)
    ]

    for pattern in version_patterns:
        matches = re.findall(pattern, name_rest)
        if matches:
            # Return the last match (in case there are multiple)
            return int(matches[-1])

    # If no version pattern found, return 0
    return 0

def extract_character_name(name_rest):
    """Extract character name from filename (first word after shot ID)"""
    # Split by underscore and get the first part as character name
    parts = name_rest.split('_')
    if parts:
        return parts[0].lower()
    return name_rest.lower()

def organize_character_files(source_dir, dest_dir):
    """Organize character files: keep only latest version per character per shot"""
    files = os.listdir(source_dir)

    # Group files by (shot_id, character_name)
    grouped = defaultdict(list)

    for filename in files:
        if not filename.endswith(('.mp4', '.mov')):
            continue

        parsed = parse_filename(filename)
        if not parsed:
            print(f"Warning: Could not parse {filename}")
            continue

        file_id = parsed['id']
        name_rest = parsed['name_rest']

        # Extract character name (first word after shot ID)
        char_name = extract_character_name(name_rest)

        # Extract version
        version = extract_version_info(name_rest, parsed['full_name'])

        # Group by (shot_id, character_name)
        key = (file_id, char_name)
        grouped[key].append({
            'filename': filename,
            'version': version,
            'parsed': parsed
        })

    # For each group, keep only the latest version
    files_to_copy = []

    for (file_id, char_name), file_list in grouped.items():
        # Sort by version (descending) and prefer .mp4 over .mov
        file_list.sort(key=lambda x: (x['version'], x['filename'].endswith('.mp4')), reverse=True)

        # Keep the first one (highest version)
        best_file = file_list[0]

        if len(file_list) > 1:
            print(f"\nShot {file_id} - Character '{char_name}':")
            print(f"  Keeping: {best_file['filename']} (v{best_file['version']})")
            print(f"  Skipping: {[f['filename'] for f in file_list[1:]]}")

        files_to_copy.append({
            'id': file_id,
            'filename': best_file['filename'],
            'source': os.path.join(source_dir, best_file['filename'])
        })

    return files_to_copy

def organize_video_files(source_dir, dest_dir):
    """Organize video files: keep only latest version per shot ID"""
    files = os.listdir(source_dir)

    # Group files by shot_id only
    grouped = defaultdict(list)

    for filename in files:
        if not filename.endswith(('.mp4', '.mov')):
            continue

        parsed = parse_filename(filename)
        if not parsed:
            print(f"Warning: Could not parse {filename}")
            continue

        file_id = parsed['id']
        name_rest = parsed['name_rest']

        # Extract version
        version = extract_version_info(name_rest, parsed['full_name'])

        # Group by shot_id only
        grouped[file_id].append({
            'filename': filename,
            'version': version,
            'parsed': parsed
        })

    # For each shot, keep only the latest version
    files_to_copy = []

    for file_id, file_list in grouped.items():
        # Sort by version (descending) and prefer .mp4 over .mov
        file_list.sort(key=lambda x: (x['version'], x['filename'].endswith('.mp4')), reverse=True)

        # Keep the first one (highest version)
        best_file = file_list[0]

        if len(file_list) > 1:
            print(f"\nShot {file_id}:")
            print(f"  Keeping: {best_file['filename']} (v{best_file['version']})")
            print(f"  Skipping: {[f['filename'] for f in file_list[1:]]}")

        files_to_copy.append({
            'id': file_id,
            'filename': best_file['filename'],
            'source': os.path.join(source_dir, best_file['filename'])
        })

    return files_to_copy

def main():
    # Source directories
    char_source = 'f5_data_multiVACE/character_bluescreen'
    video_source = 'f5_data_multiVACE/videos'

    # Destination directory
    dest_base = 'processed_f5_data_multiVACE'

    # Remove old organized directory if it exists
    if os.path.exists(dest_base):
        print(f"Removing existing directory: {dest_base}")
        shutil.rmtree(dest_base)

    # Create destination directories
    os.makedirs(dest_base, exist_ok=True)
    char_dest_base = os.path.join(dest_base, 'character_bluescreen')
    video_dest = os.path.join(dest_base, 'videos')
    os.makedirs(video_dest, exist_ok=True)

    print("=" * 80)
    print("ORGANIZING CHARACTER BLUESCREEN FILES")
    print("(Keeping only latest version per character per shot)")
    print("=" * 80)

    # Process character files
    char_files = organize_character_files(char_source, char_dest_base)

    # Copy character files to organized structure
    for file_info in char_files:
        file_id = file_info['id']
        filename = file_info['filename']
        source_path = file_info['source']

        # Create ID-specific folder
        id_folder = os.path.join(char_dest_base, file_id)
        os.makedirs(id_folder, exist_ok=True)

        # Copy file
        dest_path = os.path.join(id_folder, filename)
        shutil.copy2(source_path, dest_path)

    print(f"\n✓ Copied {len(char_files)} character files")

    print("\n" + "=" * 80)
    print("ORGANIZING VIDEO FILES")
    print("(Keeping only latest version per shot)")
    print("=" * 80)

    # Process video files
    video_files = organize_video_files(video_source, video_dest)

    # Copy video files
    for file_info in video_files:
        filename = file_info['filename']
        source_path = file_info['source']
        dest_path = os.path.join(video_dest, filename)
        shutil.copy2(source_path, dest_path)

    print(f"\n✓ Copied {len(video_files)} video files")

    print("\n" + "=" * 80)
    print("ORGANIZATION COMPLETE!")
    print("=" * 80)
    print(f"Organized dataset saved to: {dest_base}/")
    print(f"  - Character files: {len(char_files)} files in {char_dest_base}/")
    print(f"  - Video files: {len(video_files)} files in {video_dest}/")

if __name__ == '__main__':
    main()
