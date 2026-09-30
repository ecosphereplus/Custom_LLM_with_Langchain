import unittest
from semif import SemIfClassifier

class TestSemIfClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = SemIfClassifier(model_name_or_path="openai-community/gpt2")

    def test_classify_choice_structure(self):
        state = "The system is experiencing high memory usage and latency."
        options = ["INFRASTRUCTURE", "DATABASE", "FRONTEND"]
        result = self.classifier.classify_choice(state=state, options=options)

        self.assertIn("selected_option", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn(result["selected_option"], options)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=4)

    def test_classify_noul_structure(self):
        state = "Database disk space is at 98% capacity."
        question = "Is disk space critically low?"
        result = self.classifier.classify_noul(state=state, question=question)

        self.assertIn("is_true", result)
        self.assertIsInstance(result["is_true"], bool)
        self.assertIn("probability", result)
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)

    def test_runnable_invoke_options(self):
        input_data = {
            "state": "Payment processed successfully.",
            "options": ["SUCCESS", "FAILURE"],
            "instructions": "Determine status."
        }
        result = self.classifier.invoke(input_data)
        self.assertIn(result["selected_option"], ["SUCCESS", "FAILURE"])

    def test_runnable_invoke_question(self):
        input_data = {
            "state": "Connection timed out after 30 seconds.",
            "question": "Did the request time out?",
            "instructions": "Answer yes or no."
        }
        result = self.classifier.invoke(input_data)
        self.assertIn("is_true", result)

if __name__ == "__main__":
    unittest.main()
