from typing import List


class Announcer:
    """A class to log any messages that comes up on the developer case
    """

    messages: List[str]

    def __init__(self):
        self.messages = []

    def say(self, msg):
        print(f"📢: {msg}\n")

    def ask(self, msg):
        print(f"🤔: {msg}\n")

    def clear(self):
        self.messages = []
