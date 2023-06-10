from dotenv import load_dotenv
import os
from google.cloud import texttospeech
from google.oauth2 import service_account
from gpt import get_message

# Load the .env file
load_dotenv()


def get_mp3(input_text: str) -> None:
    client = texttospeech.TextToSpeechClient(
        credentials=service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        )
    )
    synthesis_input = texttospeech.SynthesisInput(text=input_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.2,
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    with open("output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')


if __name__ == "__main__":
    message = get_message("compliment", "jeans with a red shirt")
    get_mp3(message)
