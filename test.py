from unittest.mock import MagicMock, patch
import json
import json
from unittest.mock import MagicMock, patch

from main import (
    get_feedback, impromptu_speaking, storytelling,
    conflict_resolution_scenario, voice_interface
)




@patch("auto_app_2.Ollama")
def test_get_feedback(mock_ollama):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Scores: Clarity: 8, Tone: 7, Engagement: 9\nTips: Improve engagement with rhetorical questions."
    mock_ollama.return_value = mock_llm

    text = "This is a sample text for feedback."
    response = get_feedback(text)

    assert isinstance(response, str)
    assert "Scores:" in response
    assert "Tips:" in response


@patch("auto_app_2.Ollama")
def test_impromptu_speaking(mock_ollama):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Here is a short impromptu speech about climate change."
    mock_ollama.return_value = mock_llm

    topic = "Climate Change"
    response = impromptu_speaking(topic)

    assert isinstance(response, str)
    assert "climate change" in response.lower()


@patch("auto_app_2.process_feedback")
def test_storytelling(mock_process_feedback):
    mock_process_feedback.return_value = ("Good story structure.", '{"Clarity": 9}', "Add more descriptive language.")

    story = "Once upon a time..."
    feedback, scores, tips = storytelling(story)

    assert "Good story structure." in feedback
    assert "Clarity" in json.loads(scores)
    assert "Add more descriptive language." in tips


@patch("auto_app_2.Ollama")
def test_conflict_resolution_scenario(mock_ollama):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "I understand your frustration. Let's work together to resolve this."
    mock_ollama.return_value = mock_llm

    response = conflict_resolution_scenario()

    assert isinstance(response, str)
    assert "Let's work together" in response


@patch("auto_app_2.transcribe_audio")
@patch("auto_app_2.process_feedback")
def test_voice_interface(mock_process_feedback, mock_transcribe_audio):
    mock_transcribe_audio.return_value = "This is a test audio transcription."
    mock_process_feedback.return_value = ("Feedback content.", '{"Clarity": 9}', "Tips on speaking.")

    audio_path = "test.wav"
    transcription, feedback, scores, tips = voice_interface(audio_path)

    assert transcription == "This is a test audio transcription."
    assert "Feedback content." in feedback
    assert "Clarity" in json.loads(scores)
    assert "Tips on speaking." in tips
