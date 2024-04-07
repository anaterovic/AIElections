import os
from openai import OpenAI

class RunaAI(OpenAI):
    @property
    def auth_headers(self) -> dict[str, str]:
        return {"Subscription-Key": f"{self.api_key}"}

RUNAAI_SUBSCRIPTION_KEY = os.environ.get("RUNAAI_SUBSCRIPTION_KEY")

client = RunaAI(
    api_key=RUNAAI_SUBSCRIPTION_KEY,
    base_url='https://api.runaai.com/v1/',
)
