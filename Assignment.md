# Lab 1 Assignment

## Task 1: LLM Serving

Based on the `eval_llm.py` script, please create a new script `test_llm_throughput.py` to test the throughput of the LLM model under different concurrency levels. 

Run the script and plot the throughput curve. Submit your script and the throughput curve as `llm_throughput.png`.

## Task 2: ASR Serving

Run `eval_asr.py` script with your own audio clip and report the transcription result. Submit your audio clip file as `asr.wav` and the transcription result as `asr.txt`.

## Task 3: TTS Serving

Run `eval_tts.py` script with your own voice. Raplace the reference audio and reference text and change the generate text. Submit your reference audio file as `clone.wav` and reference text as `clone.txt`. Also submit your generate text as `tts.txt` and the generated audio file as `tts.wav`.

# Submission

Please submit your code and results to Blackboard by uploading the following files:
```
test_llm_throughput.py
llm_throughput.png
asr.wav
asr.txt
clone.wav
clone.txt
tts.txt
tts.wav
```