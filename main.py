from src.data_preproc import DataPreprocessor
from src.original_model import StackingModel
from src.original_test import ModelEvaluator
from src.alternative_data_preproc import AlternativeDataPreprocessor
from src.alternative_model import AlternativeStackingModel
from src.alternative_test import AlternativeModelEvaluator
from src.fake_model import FakeModel
from src.llm import LLMAnalyzer
from src.llm_analysis import LLMAnalysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run different models and analyses.")
    parser.add_argument('--input_file', type=str, default="input/ground collapse.xlsx",
                        help="Input file for data preprocessing.")
    parser.add_argument('--input_prompt_pdf', type=str, default="input/prompt.pdf",
                        help="Input file with LLM prompt.")
    parser.add_argument('--llm_repetitions', type=int, default=10,
                        help='How many times run the same LLM for experiment')
    parser.add_argument('--llm_models', type=str, nargs='+',
                        default=["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4.5-preview"],
                        help='List of LLM models to use')
    parser.add_argument('--llm_temperatures', type=float, nargs='+', default=[0, 0.5, 0.8, 1.0, 1.5],
                        help='List of LLM temperatures to use')
    args = parser.parse_args()

    preprocessor = DataPreprocessor(args.input_file)
    preprocessor.split_data()
    preprocessor.filter_collapse_data()
    preprocessor.create_train_set()
    preprocessor.save_data("input/original_train.csv", "input/original_test.csv")

    original_model = StackingModel(input_file=r"input/original_train.csv")
    original_model.train_and_evaluate()
    original_model.save_model(model_filename=r'output/original_stacked_model.pkl')

    original_evaluator = ModelEvaluator(
        model_filename=r'output/original_stacked_model.pkl',
        test_file=r"input/original_test.csv"
    )
    original_evaluator.evaluate(
        roc_curve_filename=r'output/original_roc_curve.png',
        result_excel_file=r'output/original_predicted_results.csv'
    )

    alternative_preprocessor = AlternativeDataPreprocessor(args.input_file)
    alternative_preprocessor.split_data()
    alternative_preprocessor.save_data("input/alternative_train.csv", "input/alternative_test.csv")

    alternative_model = AlternativeStackingModel(input_file=r"input/alternative_train.csv")
    alternative_model.save_scaler(scaler_filename=r'output/scaler.pkl')
    alternative_model.tune_hyperparameters()
    alternative_model.train_and_evaluate()
    alternative_model.save_model(model_filename=r'output/alternative_stacked_model.pkl')
    alternative_model.print_feature_importances()

    alternative_evaluator = AlternativeModelEvaluator(
        model_filename=r'output/alternative_stacked_model.pkl',
        test_file=r"input/alternative_test.csv",
        scaler_filename=r'output/scaler.pkl'
    )
    alternative_evaluator.evaluate(roc_curve_filename=r'output/alternative_roc_curve.png',
                                   result_excel_file=r'output/alternative_predicted_results.csv')

    fake_model = FakeModel(r"input/original_train.csv", r"input/original_test.csv")
    print("ROC-AUC score:", fake_model.merge_and_evaluate())

    #llm_temperatures = [temp for temp in args.llm_temperatures if temp != 0] + [temp for temp in args.llm_temperatures if temp != 0] * (args.llm_repetitions - 1)
    #llm_analyzer = LLMAnalyzer(args.llm_models, llm_temperatures, args.input_prompt_pdf, "output/llms.csv")
    #llm_analyzer.analyze()
    #llm_analyzer.save_results()
#
    #llm_analysis = LLMAnalysis(
    #    input_file="output/llms.csv",
    #    output_agg_file="output/llm_agg_results.tex",
    #    output_plot_file="output/llm_ground_collapse_plot.png"
    #)
    #llm_analysis.run_analysis()
