import pandas as pd
import plotnine as p9


class LLMAnalysis:
    def __init__(self, input_file: str, output_agg_file: str, output_plot_file: str):
        """
        This class is responsible for processing results from a given input CSV file,
        aggregating the results, and preparing the data for plotting.

        :param input_file: The path to the input CSV file containing LLM results.
        :type input_file: str
        :param output_agg_file: The path to the file where aggregated results will be
            saved.
        :type output_agg_file: str
        :param output_plot_file: The path to the file where the generated plot will
            be saved.
        :type output_plot_file: str
        """
        self.input_file = input_file
        self.output_agg_file = output_agg_file
        self.output_plot_file = output_plot_file
        self.llm_results = pd.read_csv(input_file)
        self.agg_results = None

    def aggregate_results(self):
        self.agg_results = self.llm_results.groupby(['model'], as_index=False).agg({
            'ground_collapse': ['mean', 'std', 'min', 'max', 'count'],
            'ground_subsidence': ['mean', 'std', 'min', 'max', 'count']
        })

    def save_aggregated_results(self):
        with open(self.output_agg_file, 'w') as f:
            f.write(self.agg_results.to_latex(index=False, float_format="%.2f"))

    def plot_results(self):
        self.llm_results['temperature'] = self.llm_results['temperature'].astype('str')
        plot = (p9.ggplot(self.llm_results, p9.aes(x='temperature', y='ground_collapse', color='model'))
                + p9.geom_boxplot()
                + p9.labs(title='Ground Collapse Weight by Model and Temperature', x='Temperature', y='Ground Collapse Weight')
                + p9.theme_minimal()
                + p9.theme(legend_position='none')
                + p9.facet_wrap('~model', nrow=1))
        plot.save(self.output_plot_file, width=14, height=4.5)

    def run_analysis(self):
        self.aggregate_results()
        self.save_aggregated_results()
        self.plot_results()


if __name__ == "__main__":
    analysis = LLMAnalysis(
        input_file="../output/llms.csv",
        output_agg_file="../output/llm_agg_results.tex",
        output_plot_file="../output/llm_ground_collapse_plot.png"
    )
    analysis.run_analysis()