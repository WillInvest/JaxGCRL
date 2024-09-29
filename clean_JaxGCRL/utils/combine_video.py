import os
import re
import subprocess

def extract_index(filename):
    # Use regular expression to find the number at the end of the filename before the extension
    match = re.search(r'snake_game_(\d+)\.avi', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # Return a large number if no match is found, to push such files to the end

def combine_videos(output_file: str, root_dir: str, pattern: str):
    # Find all .avi files in the root directory matching the pattern
    files = [f for f in os.listdir(root_dir) if f.startswith(pattern) and f.startswith("snake_game") and f.endswith(".avi")]
    
    # Sort the files by the index extracted from the filename
    files.sort(key=extract_index)
    
    # Create a temporary text file with the list of videos
    with open("input_videos.txt", "w") as f:
        for file in files:
            full_path = os.path.join(root_dir, file)
            f.write(f"file '{full_path}'\n")  # Write absolute path
    
    # Use ffmpeg to concatenate the videos
    command = [
        "ffmpeg",
        "-f", "concat",                # Concatenate mode
        "-safe", "0",                  # Allow absolute paths
        "-i", "input_videos.txt",      # The list of videos to concatenate
        "-c", "copy",                  # Copy the codec to avoid re-encoding
        "-y",                          # Automatically overwrite the output file
        output_file                    # The output file name
    ]
    
    try:
        # Suppress output except for the last line
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Combined videos saved to {output_file}")

        # After successfully combining, delete all the source .avi files
        for file in files:
            os.remove(os.path.join(root_dir, file))
        print(f"Deleted {len(files)} source video files.")
        
    except subprocess.CalledProcessError as e:
        print("Error during video concatenation:", e)
    finally:
        # Clean up the temporary file
        if os.path.exists("input_videos.txt"):
            os.remove("input_videos.txt")
