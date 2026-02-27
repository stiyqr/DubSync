import cv2
from IPython.display import clear_output
import subprocess
import os
from datetime import datetime, timedelta

def get_video_duration(video_filename):
    video = cv2.VideoCapture(video_filename)

    # Get the frame count and frame rate (FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Calculate duration (in seconds)
    if fps > 0:
        duration = frame_count / fps
    else:
        duration = 0

    video.release()
    return duration

def cut_video_by_speaker(video_file, segments, speakers):
    i = 0
    cut_files = []
    previous_end_time = datetime.strptime('00:00:00.000', '%H:%M:%S.%f').time()
    previous_end_time = datetime.combine(datetime.min, previous_end_time)
    total_duration = get_video_duration(video_file)

    for segment in segments:
        segment_start_time = datetime.strptime(segment['start_time'], '%H:%M:%S.%f').time()
        segment_end_time = datetime.strptime(segment['end_time'], '%H:%M:%S.%f').time()
        segment_start_time = datetime.combine(datetime.min, segment_start_time)
        segment_end_time = datetime.combine(datetime.min, segment_end_time)
        print(segment['speaker'], "start =", segment_start_time.strftime('%H:%M:%S.%f'), "end =", segment_end_time)
        print(segment['text'])

        # If there is a gap before this segment, cut that as "no speaker"
        if segment_start_time > previous_end_time:
            silent_segment_start = previous_end_time
            silent_segment_end = segment_start_time
            silent_segment_duration = silent_segment_end - silent_segment_start
            # locale.getpreferredencoding = locale.getpreferredencoding
            silent_output_file = f"/content/cut_video{i}.mp4"
            silent_video_cut_command = f"ffmpeg -y -i {video_file} -ss {silent_segment_start.strftime('%H:%M:%S.%f')} -t {silent_segment_duration} -async 1 {silent_output_file}"
            subprocess.run(silent_video_cut_command, shell=True)
            cut_files.append((silent_output_file, silent_output_file, "no_speaker", ""))
            i += 1
            clear_output()

        # Cut the current speaker segment
        segment_duration = segment_end_time - segment_start_time
        # locale.getpreferredencoding = locale.getpreferredencoding
        output_file = f"/content/cut_video{i}.mp4"
        video_cut_command = f"ffmpeg -y -i {video_file} -ss {segment_start_time.strftime('%H:%M:%S.%f')} -t {segment_duration} -async 1 {output_file}"
        subprocess.run(video_cut_command, shell=True)
        cut_files.append((output_file, output_file, segment['speaker'], segment['text']))
        #===============================
        speaker = segment['speaker']
        if speaker not in speakers:
            speakers.append(speaker)
        print(speakers)
        #==============================

        i += 1
        # print(output_file, "video files cut.")
        # print()
        clear_output()

        # Update the previous end time to the current segment's end time
        previous_end_time = segment_end_time

    # # If there's time left after the last segment, add a "no speaker" segment
    # total_duration_timedelta = timedelta(seconds=total_duration)
    # total_duration_datetime = datetime.min + total_duration_timedelta
    # if previous_end_time < total_duration_datetime and (total_duration_datetime - previous_end_time) > timedelta(milliseconds=5):
    #     silent_segment_start = previous_end_time
    #     silent_segment_duration = total_duration_datetime - previous_end_time
    #     silent_output_file = f"/content/cut_video{i}_end.mp4"
    #     silent_video_cut_command = f"ffmpeg -y -i {video_file} -ss {silent_segment_start.strftime('%H:%M:%S.%f')} -t {silent_segment_duration} -async 1 {silent_output_file}"
    #     subprocess.run(silent_video_cut_command, shell=True)
    #     cut_files.append((silent_output_file, silent_output_file, "no_speaker", ""))
    #     print(silent_output_file, "silent video cut.")
    #     clear_output()
    return cut_files

def stitch_video_cuts(cut_files):
    with open("input.txt", "w") as file:
      for cut_file in cut_files:
          file.write(f"file '{cut_file}'\n")
          # print(f"file '{cut_file}'")
    result_file = "output_video.mp4"
    if os.path.exists(result_file):
        os.remove(result_file)
    concat_command = "ffmpeg -f concat -safe 0 -i input.txt -c copy output_video.mp4"
    try:
        result = subprocess.run(concat_command, shell=True, capture_output=True, check=True)
        print("Concatenation successful!")
    except subprocess.CalledProcessError as e:
        print("Concatenation failed:", e.stderr.decode('utf-8'))
    clear_output()
    # files.download(result_file)
    return result_file
