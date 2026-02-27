from whisper_nemo_config import *
from whisper_nemo_utils import *

def extract_audio_timestamps(audio_file):
    if enable_stemming:
        vocal_target = remove_music(audio_file)
    else:
        vocal_target = audio_file

    word_timestamps, audio_waveform, info = transcript_audio(vocal_target)
    temp_path = convert_audio(audio_waveform)
    wsm, speaker_ts = diarize_speakers(word_timestamps, temp_path)
    wsm, ssm = add_punctuation(wsm, speaker_ts, info)
    create_transcript_files(audio_file, ssm)
    cleanup(temp_path)

    return ssm

def remove_music(audio_path):
    # Isolate vocals from the rest of the audio
    return_code = os.system(
        f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "temp_outputs"'
    )

    if return_code != 0:
        logging.warning("Source splitting failed, using original audio file.")
        vocal_target = audio_path
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(audio_path))[0],
            "vocals.wav",
        )

    return vocal_target

def transcript_audio(vocal_target):
    whisper_model = faster_whisper.WhisperModel(
        whisper_model_name, device=device, compute_type=compute_type
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    audio_waveform = faster_whisper.decode_audio(vocal_target)
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if suppress_numerals
        else [-1]
    )

    if batch_size > 0:
        transcript_segments, info = whisper_pipeline.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            batch_size=batch_size,
            without_timestamps=True,
        )
    else:
        transcript_segments, info = whisper_model.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            without_timestamps=True,
            vad_filter=True,
        )

    full_transcript = "".join(segment.text for segment in transcript_segments)

    # clear gpu vram
    del whisper_model, whisper_pipeline
    torch.cuda.empty_cache()

    # align the transcription with the original audio using Forced Alignment
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # audio_waveform = audio_waveform.to(alignment_model.dtype).to(alignment_model.device)
    # Convert `audio_waveform` to a PyTorch tensor before using `.to`
    audio_waveform = torch.tensor(audio_waveform).to(alignment_model.dtype).to(alignment_model.device)

    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=batch_size
    )

    del alignment_model
    torch.cuda.empty_cache()

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[info.language],
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    return word_timestamps, audio_waveform, info

def convert_audio(audio_waveform):
    # convert audio to mono for NeMo compatibility
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    torchaudio.save(
        os.path.join(temp_path, "mono_file.wav"),
        audio_waveform.cpu().unsqueeze(0).float(),
        16000,
        channels_first=True,
    )
    return temp_path

def diarize_speakers(word_timestamps, temp_path):
    # Initialize NeMo MSDD diarization model
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to("cuda")
    msdd_model.diarize()

    del msdd_model
    torch.cuda.empty_cache()

    # Reading timestamps <> Speaker Labels mapping

    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    return wsm, speaker_ts

def add_punctuation(wsm, speaker_ts, info):
    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    else:
        logging.warning(
            f"Punctuation restoration is not available for {info.language} language. Using the original punctuation."
        )

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    return wsm, ssm

def create_transcript_files(audio_path, ssm):
    with open(f"{os.path.splitext(audio_path)[0]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(f"{os.path.splitext(audio_path)[0]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

def format_segments(segments):
    for segment in segments:
        segment['start_time'] = format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker='.')
        segment['end_time'] = format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker='.')
    return segments
