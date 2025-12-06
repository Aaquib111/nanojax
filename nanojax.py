"""Small implementation of functional auto-differentiation in Python

Usage:
import nanojax as nj

f = lambda x: nj.sin(nj.pow(x, 2))
df_dx = grad(f)
df2_d2x = grad(df_dx)
"""

import os
from google import genai

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents="Explain how AI works in a few words",
)

print(response.text)
