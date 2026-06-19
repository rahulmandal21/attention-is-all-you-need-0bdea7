import sacrebleu
import torch
import torch.nn as nn

class MachineTranslationEvaluator:
    """
    Evaluates the performance of a machine translation model using the BLEU score.
    """

    def __init__(self, metric: str = 'bleu'):
        """
        Initializes the evaluator with the specified metric.

        Args:
            metric (str): The evaluation metric to use (default: 'bleu').
        """
        self.metric = metric

    def compute_bleu(self, predictions: list, references: list) -> float:
        """
        Computes the BLEU score for the given predictions and references.

        Args:
            predictions (list): The predicted translations.
            references (list): The reference translations.

        Returns:
            float: The BLEU score.
        """
        if self.metric == 'bleu':
            bleu = sacrebleu.corpus_bleu(predictions, [references])
            return bleu.score
        else:
            raise ValueError("Unsupported metric. Only 'bleu' is supported.")

    def evaluate(self, predictions: list, references: list) -> dict:
        """
        Evaluates the performance of the model on the given predictions and references.

        Args:
            predictions (list): The predicted translations.
            references (list): The reference translations.

        Returns:
            dict: A dictionary containing the evaluation results.
        """
        bleu_score = self.compute_bleu(predictions, references)
        return {'bleu': bleu_score}

if __name__ == "__main__":
    evaluator = MachineTranslationEvaluator()
    predictions = ["This is a test prediction", "Another test prediction"]
    references = ["This is a test reference", "Another test reference"]
    results = evaluator.evaluate(predictions, references)
    print(results)