from unstructured.partition.auto import partition
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import pandas as pd


load_dotenv()


class Weights(BaseModel):
    ground_collapse: float = Field(description="Ground collapse weight.")
    ground_subsidence: float = Field(description="Ground subsidence weight.")
    judgment: str = Field(description="Judgement of the impact of the two hazards on the basis of the provided information.")


class LLMAnalyzer:
    def __init__(self, models: list[str], temperatures: list[int], input_file: str, output_file: str):
        """
        Initializes an object to manage model processing with temperature settings
        and input/output file handling. This class is designed to process information
        from an input file, store results, and use a set of models and corresponding
        temperature settings for computation.

        :param models: A list of model names to be used for processing.
        :type models: list[str]
        :param temperatures: A list of integer temperature values corresponding to each model.
        :type temperatures: list[int]
        :param input_file: The input file path containing data to be processed.
        :type input_file: str
        :param output_file: The output file path for storing processed results.
        :type output_file: str
        """
        self.models = models
        self.temperatures = temperatures
        self.input_file = input_file
        self.output_file = output_file
        self.results = []
        self.big_prompt = self.load_prompt(input_file)

    @staticmethod
    def load_prompt(input_file: str) -> str:
        elements = partition(input_file)
        return "\n\n".join([str(el) for el in elements])

    def analyze(self):
        for model_name in self.models:
            for temp in self.temperatures:
                try:
                    model = ChatOpenAI(model=model_name, temperature=temp)
                    structured_model = model.with_structured_output(Weights)
                    result = structured_model.invoke(self.big_prompt)
                    self.results.append({
                        "model": model_name,
                        "temperature": temp,
                        "ground_collapse": result.ground_collapse,
                        "ground_subsidence": result.ground_subsidence,
                        "judgment": result.judgment
                    })
                except Exception as e:
                    print(e)

    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_file, index=False)

if __name__ == "__main__":
    models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4.5-preview"]
    temperatures = [0] + [0.5, 0.8, 1.0, 1.5] * 10
    analyzer = LLMAnalyzer(models, temperatures, "../input/description.pdf", "output/llms.csv")
    analyzer.analyze()
    analyzer.save_results()
