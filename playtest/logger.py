from typing import List


class Announcer:
    """A class to log any messages that comes up on the developer case
    """

    messages: List[str]
    verbose: bool

    def __init__(self, verbose=True):
        self.messages = []
        self.verbose = verbose

    def say(self, msg):
        if self.verbose:
            print(f"📢: {msg}\n")

    def ask(self, msg):
        if self.verbose:
            print(f"🤔: {msg}\n")

    def clear(self):
        self.messages = []
