from whisper_nemo import *
from wav2lip import *

def processing_whisper(video_file, audio_file, face_images):
    print("Files downloaded")
    speakers = []

    # # find number of speakers
    # speaker_count = get_speaker_count(face_images)

    segments = extract_audio_timestamps(audio_file)
    segments = format_segments(segments)
    cut_files = cut_video_by_speaker(video_file, segments, speakers)
    print("Video files cut")

    return segments, cut_files, speakers

def processing_wav2lip(cut_files, face_images, speaker_image_map):
    print(speaker_image_map)
    print()
    print(cut_files)

    # process cut files using wav2lip
    processed_cuts = wav2lip_process_list(cut_files, face_images, speaker_image_map)

    # stitch processed files*
    stitched_file = stitch_video_cuts(processed_cuts)
    print("Video stitched")

    return stitched_file
