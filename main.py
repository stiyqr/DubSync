import gradio as gr
import pandas as pd
import soundfile as sf
import os
import shutil
from datetime import datetime

from processing import *
from wav2lip import get_speaker_set

#=========================================#
#            Processing Video             #
#=========================================#
class VideoProcessor:
    def __init__(self):
        self.segments = None
        self.cut_files = None
        self.speakers = None
        self.actor_images = None
        self.image_name_map = {}
        self.max_speakers = 10


    #=========================================#
    #           to process whixperx           #
    #=========================================#
    def process_video_step1(self, video, audio, image):
        if video is None and audio is None and image is None:
            raise gr.Warning("Please upload a video, audio and image file")
        elif video is None:
            raise gr.Warning("Please upload a video file")
        elif audio is None:
            raise gr.Warning("Please upload an audio file")
        elif image is None:
            raise gr.Warning("Please upload at least an image file")

        # create new folder
        input_dir = "input_folder"
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

        output_dir = "processed_video"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_video_filename = os.path.basename(video)
        input_audio_filename = os.path.basename(audio)
        output_filename = f"{timestamp}.mp4"

        input_video_filepath = os.path.join(input_dir, input_video_filename)
        input_audio_filepath = os.path.join(input_dir, input_audio_filename)
        output_filepath = os.path.join(output_dir, output_filename)

        # save the audio data to a file
        os.rename(video, input_video_filepath)
        os.rename(audio, input_audio_filepath)

        # save the images data to file
        self.actor_images = []
        for i, img_tuple in enumerate(image):
            img_data, label = img_tuple
            input_image_filename = os.path.basename(img_data)
            input_image_filepath = os.path.join(input_dir, input_image_filename)
            self.actor_images.append(input_image_filepath)

            os.rename(img_data, input_image_filepath)

        self.segments, self.cut_files, self.speakers = processing_whisper(input_video_filepath, input_audio_filepath, self.actor_images)

        script_text = ""
        for cut_file in self.cut_files:
            if cut_file[2] != "no_speaker":
                script_text += f"{cut_file[2]} : {cut_file[3]}\n"

        speakers_assign = self.assign_speaker_image(self.speakers, self.actor_images)

        return script_text, *speakers_assign, gr.update(interactive = True), gr.update(interactive = True)


    #=========================================#
    #       assigning image to speaker        #
    #=========================================#
    def assign_speaker_image(self, speakers, actor_images):
        updates = []

        # get the image name by get_speaker_set
        image_name_set = get_speaker_set(self.actor_images)
        image_names = list(image_name_set)

        for index, speaker in enumerate(speakers):
            if speaker is not None:
                # For the first dropdown, use "Please select" as the label
                if index == 0:
                    label = "Please select actor image to assign to the speaker"
                else:
                    label = " "

                updates.append(gr.update(choices=image_names,
                                        label=label,
                                        visible=True,
                                        interactive=True,
                                        info=speaker))
            else:
                updates.append(gr.update(visible=False))

        while len(updates) < self.max_speakers:
            updates.append(gr.update(visible=False))

        return updates


    #=========================================#
    #              confirm edits              #
    #=========================================#
    def edit_script_speaker(self, script_text):
        #===== for updated script_text to update cut_files =====#
        updated_cut_files = []
        lines = script_text.strip().split('\n')
        cut_index = 0  # Separate index for `self.cut_files`

        for line in lines:
            if ":" in line and cut_index < len(self.cut_files):  # Format in "speaker: text"
                speaker, text = line.split(": ", 1)
                cut_file = self.cut_files[cut_index]

                # If no_speaker is in the cut_file, add it first
                if cut_file[2] == "no_speaker":
                    updated_cut_files.append(cut_file)  # Add the original "no_speaker" entry
                    cut_index += 1  # Move to the next segment in self.cut_files

                # Add the speaker entry
                if cut_index < len(self.cut_files):  # Check again after increment
                    cut_file = self.cut_files[cut_index]
                    updated_cut_files.append((cut_file[0], cut_file[1], speaker.strip(), text.strip()))
                    cut_index += 1  # Move to the next segment in self.cut_files

        # Ensure any remaining segments are added after the loop
        while cut_index < len(self.cut_files):
            updated_cut_files.append(self.cut_files[cut_index])
            cut_index += 1

        self.cut_files = updated_cut_files

        script_text = ""
        for cut_file in self.cut_files:
            if cut_file[2] != "no_speaker":
                script_text += f"{cut_file[2]} : {cut_file[3]}\n"
                if cut_file[2] not in self.speakers:
                        self.speakers.append(cut_file[2])

        speakers_assign = self.assign_speaker_image(self.speakers, self.actor_images)

        return script_text, *speakers_assign


    #=========================================#
    #           to process wav2lip            #
    #=========================================#
    def process_video_step2(self, *choice_image):
        #===== images choices =====#
        selected_images = []
        for image_name in choice_image:
            if image_name:
                selected_images.append(image_name)

        speaker_image_map = dict(zip(self.speakers, selected_images))  # map speaker to image name


        result = processing_wav2lip(self.cut_files, self.actor_images, speaker_image_map)

        return result

    #=========================================#
    #             print cut_files             #
    #=========================================#
    def print_cut_files(self):
        for cut_file in self.cut_files:
            print(f"[{cut_file[0]}] [{cut_file[1]}], [{cut_file[2]}], [{cut_file[3]}]")


#=========================================#
#    Showing all the processed video      #
#=========================================#
def get_all_processed_videos():
    output_dir = "processed_video"
    processed_videos = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.mp4')]

    # Ensure the list has exactly 10 elements
    update_show = processed_videos + [None] * (10 - len(processed_videos))

    # Update the btn_list with video paths
    updates = []
    for video_path in update_show:
        if video_path is not None:
            filename = os.path.basename(video_path)
            updates.append(gr.update(value=video_path, label=filename, visible=True))
        else:
            updates.append(gr.update(visible=False))

    return updates

video_list = []


#=========================================#
#            Custom Interface             #
#=========================================#
def get_custom_blocks():
    theme_color = gr.themes.Default(primary_hue="lime")
    title = "DubSync"
    css = """footer{display:none !important}
          .group-color {background-color: #27272A;}
    """
    #""".gradio-container {margin: 0 !important};"""

    return {"theme": theme_color, "title": title, "css": css}


#=========================================#
#          Creating the interface         #
#=========================================#
custom_block = get_custom_blocks()
theme = custom_block["theme"]
title = custom_block["title"]
css = custom_block["css"]

with gr.Blocks(theme = theme, title = title, css = css) as create_interface:

    # Page header
    with gr.Row():
        # The title and description of the web server
        with gr.Column(scale = 12):
            with gr.Column():
                Title = gr.Markdown(
                    f"""<h1 style="color: #7ABF13;">{title}</h1>"""
                )
                Desc = gr.Markdown(
                    """
                    Process your video to match the dubbing audio.
                    """
                )

        #space purpose
        with gr.Column(scale = 5):
            pass

    # Space purpose
    with gr.Row():
        pass


    processing = VideoProcessor()
    with gr.Row(equal_height=True):
        # Input Column
        with gr.Column():
            video_input = gr.Video(label="Input Video Upload", sources="upload")
            audio_input = gr.Audio(label="Input Audio Upload", type="filepath", sources="upload")
            image_input = gr.Gallery(label="Input Image Upload (select all images)", height="250px")

        # Script Text and Assign Image Column
        with gr.Column():
            with gr.Group(elem_classes="group-color"):
                max_speakers = 10
                script_list = gr.Textbox(label="Script Text", interactive=True)
                speaker_list = [gr.Dropdown(visible=False, interactive=True) for _ in range(max_speakers)]


        # Output Column
        with gr.Column():
            with gr.Group(elem_classes="group-color"):
                processed_video_output = gr.Video(label="Video Output", show_download_button=True, autoplay=False)
                with gr.Row():
                    for i in range(7):
                        btn = gr.Video(visible=False)
                        video_list.append(btn)


    # Button
    with gr.Row(equal_height = True):
        with gr.Column():
            submit_button = gr.Button("Submit", variant = "primary")
        with gr.Column():
            confirm_edits_button = gr.Button("Confirm Edits", variant = "secondary", interactive=False)
            finalize_button = gr.Button("Finalize Processing", variant = "primary", interactive=False)
        with gr.Column():
            show_all_output = gr.Button("Show All Processed Output")


    # Button action
    submit_button.click(
        fn=processing.process_video_step1,
        inputs=[video_input, audio_input, image_input],
        outputs=[script_list, *speaker_list, confirm_edits_button, finalize_button],
    )

    confirm_edits_button.click(
        fn=processing.edit_script_speaker,
        inputs=[script_list],
        outputs=[script_list, *speaker_list],
    )

    finalize_button.click(
        fn=processing.process_video_step2,
        inputs=[*speaker_list],
        outputs=processed_video_output,
    )

    show_all_output.click(
        fn=processing.print_cut_files,
        inputs=None,
        outputs=None
    )

    # show_all_output.click(
    #     fn=get_all_processed_videos,
    #     inputs=None,
    #     outputs=video_list
    # )


#=========================================#
#           Launch the interface          #
#=========================================#
if __name__ == "__main__":
    create_interface.launch(share=True, debug=True)
