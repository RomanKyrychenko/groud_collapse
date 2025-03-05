from unstructured.partition.auto import partition
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import pandas as pd

load_dotenv()

models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4.5-preview"]
temperatures = [0] + [0.5, 0.8, 1.0, 1.5]*10
results = []

class Weights(BaseModel):
    ground_collapse: float = Field(description="Ground collapse weight.")
    ground_subsidence: float = Field(description="Ground subsidence weight.")
    judgment: str = Field(description="Judgement of the impact of the two hazards on the basis of the provided information.")

elements = partition("input/description.pdf")
big_prompt = "\n\n".join([str(el) for el in elements])

for model_name in models:
    for temp in temperatures:
        try:
            model = ChatOpenAI(model=model_name, temperature=temp)
            structured_model = model.with_structured_output(Weights)
            result = structured_model.invoke(big_prompt)
            results.append({
                "model": model_name,
                "temperature": temp,
                "ground_collapse": result.ground_collapse,
                "ground_subsidence": result.ground_subsidence,
                "judgment": result.judgment
            })
        except Exception as e:
            print(e)

df = pd.DataFrame(results)

# Save results to a file
df.to_csv('output/llms.csv', index=False)
