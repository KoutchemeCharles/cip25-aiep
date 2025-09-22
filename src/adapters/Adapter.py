from dspy.adapters.chat_adapter import ChatAdapter

class NoSystemPromptAdapter(ChatAdapter):
    def format(self, signature, demos, inputs):
        messages = super().format(signature, demos, inputs)
        # Remove the system prompt (first message)
        if messages and messages[0]["role"] == "system":
            messages = messages[1:]
        return messages
