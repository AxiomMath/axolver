from abc import ABC, abstractmethod


class Generator(ABC):
    @abstractmethod
    def generate(self, rng, is_train):
        """
        Returns (problem, question, answer) where question can be None.
        problem and answer are required raw Python objects.
        """
        pass

    def encode_class_id(self, problem_data, question_data, answer_data):
        # Override this method if you want more granular metrics
        return 0

    @abstractmethod
    def evaluate(self, problem, question, answer, hyp, metrics):
        """
        Evaluate a hypothesis against the expected answer.
        Returns metrics_dict where metrics_dict["is_valid"] is always present.
        is_valid: 1 if correct, 0 if incorrect, -1 if decoding error.
        metrics: list of metric names to compute, or None.
        """
        pass
