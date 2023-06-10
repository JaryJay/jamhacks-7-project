from dotenv import load_dotenv
import os
import openai

# load the .env file
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


def get_message(mode: str, description: str) -> str:
    """Get a message from GPT-3.5-Turbo.

    :param mode: The mode of the message, either "compliment" or "insult".
    :param description: The description of the person to give the message to.
    :return: The message from GPT-3.5-Turbo.
    === Example Usage ===
    >>> get_message("compliment", "jeans with a red shirt")
    "You are a nice person who likes to give compliments."
    >>> get_message("insult", "jeans with a red shirt")
    "You are a mean person who likes to give insults."
    """

    system_msg = (
        "You are a nice person who likes to give compliments."
        if mode == "compliment"
        else "You are a mean person who likes to give insults."
    )
    user_msg = f"Give a one sentence {mode} to somebody wearing {description}."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    message = get_message("compliment", "jeans with a red shirt")
    # message = get_message("insult", "jeans with a red shirt")
    print(message)
