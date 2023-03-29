from prompt import Prompt

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatGPT:

  def __init__(self):
    self.prompt = Prompt()
    self.model = os.getenv("OPENAI_MODEL", default="text-davinci-003")
    self.temperature = float(os.getenv("OPENAI_TEMPERATURE", default=0))
    self.frequency_penalty = float(
      os.getenv("OPENAI_FREQUENCY_PENALTY", default=0))
    self.presence_penalty = float(
      os.getenv("OPENAI_PRESENCE_PENALTY", default=0.6))
    self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", default=240))
    self.top_p = int(os.getenv("TOP_P", default=1.0))
    
  def get_response(self):
    response = openai.Completion.create(
      model=self.model,
      prompt=self.prompt.generate_prompt(),
      temperature=self.temperature,
      frequency_penalty=self.frequency_penalty,
      presence_penalty=self.presence_penalty,
      max_tokens=self.max_tokens)
    return response['choices'][0]['text'].strip()

  def add_msg(self, text):
    self.prompt.add_msg(text)

  def response_explain_code(self) -> str:
    response = openai.Completion.create(
      model=self.model,
      prompt=self.prompt.generate_explain_code_prompt(),
      temperature=self.temperature,
      top_p=self.top_p,
      frequency_penalty=self.frequency_penalty,
      presence_penalty=0.0,
      max_tokens=self.max_tokens,
      stop=["\"\"\""])
    return response['choices'][0]['text'].strip()

