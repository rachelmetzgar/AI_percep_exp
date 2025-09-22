"""
Module: llm_utils.py

Purpose:
    Utilities for transcription (Whisper) and conversational agent responses
    (OpenAI API) in the AI Perception experiment.

Functions:
    - transcribe_speech: Transcribes recorded audio with Whisper.
    - get_llm_response:  Generates an LLM response given user input text.
    - get_response_and_display: Full pipeline of transcription -> LLM response -> TTS playback.

Inputs:
    - audio_file          : Path to audio segment file for transcription.
    - whisper_model       : Pre-loaded Whisper model instance.
    - preprocess_audio_fn : Function to preprocess audio (mono, resample).
    - client              : OpenAI client.
    - system_prompt       : String system prompt for conversation.
    - temperature         : LLM sampling temperature.
    - conversation_history: List[str], maintains user/agent dialogue context.
    - thisExp             : PsychoPy ExperimentHandler (for logging).
    - trials              : PsychoPy TrialHandler (for timestamps + metadata).
    - win                 : PsychoPy Window (for visual feedback).
    - audio_dir           : Path to subject audio directory.
    - log_event_time_fn   : Function to log timestamps (event, globalClock, trials).

Outputs:
    - Transcribed text (string).
    - LLM response (string).
    - Saved audio files (LLM responses, WAV).
    - Updates to PsychoPy data logs.

Usage:
    from utils.llm_utils import transcribe_speech, get_llm_response, get_response_and_display

Author: Rachel C. Metzgar
Date: 2025-02-12    
"""

import os
import soundfile as sf
import sounddevice as sd
from psychopy import visual, core


def transcribe_speech(audio_file: str, whisper_model, preprocess_audio_fn) -> str:
    """Transcribe the given audio file using Whisper."""
    print(f"Transcribing audio file: {audio_file}")
    processed_audio = preprocess_audio_fn(audio_file, "processed_audio.wav")
    result = whisper_model.transcribe(processed_audio)
    transcription = result["text"]
    print(f"Transcription result: {transcription}")
    return transcription


def get_llm_response(agent: str, text: str, temperature: float,
                     client, system_prompt: str,
                     conversation_history: list, thisExp) -> str:
    """Get the LLM response for the given agent and user text."""
    conversation_history.append(f"User: {text}")

    # Keep only last ~5 turns (10 entries: user + agent)
    last_5_interactions = conversation_history[-10:]
    prompt_text = "\n".join(last_5_interactions)

    print("Using last 5 interactions for the prompt:")
    print(prompt_text)
    print(f"Prompt length (characters): {len(prompt_text)}")

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt_text}],
            temperature=temperature,
        )
        response = chat_completion.choices[0].message.content
        conversation_history.append(f"GPT: {response}")
        print(f"LLM response: {response}")

        # Clean transcript for logging
        def clean_text(resp):
            return resp.replace("\n", " ").replace("\r", " ") if isinstance(resp, str) else resp

        llm_response = clean_text(response)
        thisExp.addData("transcript_LLM", llm_response)

        return response
    except Exception as e:
        print(f"Error during API call: {e}")
        return ""


def get_response_and_display(agent: str, segment_audio_file: str,
                             run: int, trial_num: int, topic: str, llm_int_num: int,
                             whisper_model, preprocess_audio_fn,
                             client, system_prompt: str, temperature: float,
                             conversation_history: list,
                             thisExp, trials, win, audio_dir: str,
                             log_event_time_fn, globalClock,
                             audio_captured: bool):
    """Transcribe audio, get LLM response, and save each LLM audio output."""
    try:
        if audio_captured:
            print(f"Audio captured. Attempting transcription from: {segment_audio_file}")

            if not os.path.exists(segment_audio_file):
                raise FileNotFoundError(f"Audio file not found: {segment_audio_file}")

            file_size = os.path.getsize(segment_audio_file)
            print(f"Audio file size: {file_size} bytes")

            # Transcribe
            transcribed_text = transcribe_speech(segment_audio_file, whisper_model, preprocess_audio_fn)
            if not transcribed_text or transcribed_text.strip() == "":
                raise ValueError("No transcribed text found. Transcription may have failed.")

            print(f"Transcribed Text: {transcribed_text}")
            thisExp.addData("transcript_sub", transcribed_text)

            # Get LLM response
            chatgpt_response = get_llm_response(agent, transcribed_text, temperature,
                                                client, system_prompt,
                                                conversation_history, thisExp)

            log_event_time_fn("LLM_response_start", globalClock, trials)

            # On-screen feedback: show agent identity
            agent_text_map = {
                "bot_1": "You are speaking with ChatGPT.",
                "bot_2": "You are speaking with Gemini.",
                "hum_1": "You are speaking with Casey.",
                "hum_2": "You are speaking with Sam.",
            }
            agent_text = visual.TextStim(win, text=agent_text_map.get(agent, "Unknown agent"),
                                         pos=(0, 0), height=30, color=(0, 0, 0))
            agent_text.draw()
            win.flip()
            log_event_time_fn("win_display_end", globalClock, trials)
            log_event_time_fn("response_display_start", globalClock, trials)

            # Save LLM response audio
            trials_llm_dir = os.path.join(audio_dir, "trials", "llm")
            os.makedirs(trials_llm_dir, exist_ok=True)
            llm_segment_audio_file_path = os.path.join(
                trials_llm_dir, f"run{run:02d}_{trial_num}_{topic}_{agent}_{llm_int_num}_llm.wav"
            )

            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=chatgpt_response,
            )
            with open(llm_segment_audio_file_path, "wb") as f:
                f.write(response.content)

            print(f"LLM response saved as: {llm_segment_audio_file_path}")
            trials.addData("llm_segment_name", llm_segment_audio_file_path)

            # Play back audio
            data, fs = sf.read(llm_segment_audio_file_path, dtype="float32")
            log_event_time_fn("LLM_audio_start", globalClock, trials)
            sd.play(data, fs)
            sd.wait()
            log_event_time_fn("LLM_audio_end", globalClock, trials)
            print("Audio playback finished.")

            return True  # success
        else:
            print("No significant audio captured, skipping transcription.")
            no_audio_msg = visual.TextStim(win,
                                           text="No significant audio detected. Please try again.",
                                           pos=(0, 0), height=20, color=(1, -1, -1))
            no_audio_msg.draw()
            win.flip()
            log_event_time_fn("win_display_end", globalClock, trials)
            log_event_time_fn("no_audio_display_start", globalClock, trials)
            core.wait(2)
            log_event_time_fn("no_audio_display_end", globalClock, trials)
            trials.addData("no_audio_captured", "no audio captured")
            return False
    except Exception as e:
        print(f"Error during transcription or response generation: {e}")
        error_message = visual.TextStim(win,
                                        text=f"Error: {e}",
                                        pos=(0, 0), color=(-1, -1, -1))
        error_message.draw()
        win.flip()
        return False
