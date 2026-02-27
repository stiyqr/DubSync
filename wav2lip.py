import face_recognition
from IPython.display import HTML, clear_output, Audio
from IPython.core.display import display
from base64 import b64encode
import os
import cv2
import face_recognition
import re
import subprocess
import locale

from video_utils import *

locale.getpreferredencoding = lambda: "UTF-8"

def parse_face_name(file_path):
    """Extract the name from the file path based on the assumed naming convention."""
    file_name = os.path.basename(file_path)
    name = os.path.splitext(file_name)[0]  # Remove file extension
    # Remove anything after '--', e.g., "man_1--2" becomes "man_1"
    name = re.split(r"--", name)[0]
    return name  # Assuming file name itself follows the naming rule

def get_speaker_count(face_images):
    name_set = set()
    for face_img in face_images:
        face_name = parse_face_name(face_img)
        name_set.add(face_name)

    return len(name_set)

def get_speaker_set(face_images):
    name_set = set()
    for face_img in face_images:
        face_name = parse_face_name(face_img)
        name_set.add(face_name)

    return name_set

def wav2lip_process_single_file(video_file, audio_file, speaker, face_path_list, output_vid_name):
    # Set up paths and variables for the output file
    output_file_path = f'/content/Wav2Lip/results/{output_vid_name}.mp4'

    # Delete existing output file before processing, if any
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    pad_top =  0
    pad_bottom =  10
    pad_left =  0
    pad_right =  0
    rescaleFactor =  1
    nosmooth = True

    use_hd_model = True
    checkpoint_path = 'checkpoints/wav2lip.pth' if not use_hd_model else 'checkpoints/wav2lip_gan.pth'

    if nosmooth == False:
        command = [
            "python", "inference.py",
            "--checkpoint_path", checkpoint_path,
            "--face", video_file,
            "--audio", audio_file,
            "--pads", str(pad_top), str(pad_bottom), str(pad_left), str(pad_right),
            "--resize_factor", str(rescaleFactor),
            "--image_paths", *face_path_list,
            "--speaker", speaker,
            "--outfile", output_file_path
        ]
    else:
        command = [
            "python", "inference.py",
            "--checkpoint_path", checkpoint_path,
            "--face", video_file,
            "--audio", audio_file,
            "--pads", str(pad_top), str(pad_bottom), str(pad_left), str(pad_right),
            "--resize_factor", str(rescaleFactor),
            "--nosmooth",
            "--image_paths", *face_path_list,
            "--speaker", speaker,
            "--outfile", output_file_path
        ]
    subprocess.run(command)

    #Preview output video
    if os.path.exists(output_file_path):
        clear_output()
    else:
        print("Processing failed. Output video not found.")

    return output_file_path


def wav2lip_process_list(cut_files, face_images, speaker_map):
    i = 0
    processed_cuts = []
    for cut_file in cut_files:
        video_file = cut_file[0]
        audio_file = cut_file[1]
        speaker = cut_file[2]

        if speaker == "no_speaker" or get_video_duration(video_file) < 0.1:
            processed_cuts.append(video_file)
            print('no_speaker skipped')
        else:
            speaker_name = speaker_map[speaker]
            processed_cut = wav2lip_process_single_file(video_file, audio_file, speaker_name, face_images, f"processed_vid{i}")
            processed_cuts.append(processed_cut)
        i += 1

    return processed_cuts
