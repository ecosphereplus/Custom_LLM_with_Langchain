import torch
import torch.nn.functional as F
from typing import Dict, List, Union, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.runnables import Runnable

class SemIfClassifier(Runnable):
    """
    SemIf-style System One Classifier.
    Extracts native logit probabilities directly from causal language models
    in a single forward pass without autoregressive generation.
    """

    def __init__(
        self,
        model_name_or_path: str = "openai-community/gpt2",
        device: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if model is not None:
            self.model = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)

        self.model.eval()

    def classify_choice(
        self,
        state: str,
        options: List[str],
        instructions: str = "Select the best option from the candidate choices."
    ) -> Dict[str, Any]:
        """
        Classifies the state into one of the provided options by reading logits
        directly in a single forward pass.
        """
        prompt = f"Context: {state}\nInstructions: {instructions}\nChoice:"
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        option_probs = {}
        option_log_likelihoods = {}

        with torch.no_grad():
            outputs = self.model(**prompt_inputs)
            # Logits for the next token after the prompt
            next_token_logits = outputs.logits[0, -1, :]

            for option in options:
                # Format option with leading space if needed
                opt_str = f" {option.strip()}"
                opt_token_ids = self.tokenizer.encode(opt_str, add_special_tokens=False)

                if len(opt_token_ids) == 1:
                    token_id = opt_token_ids[0]
                    log_prob = F.log_softmax(next_token_logits, dim=-1)[token_id].item()
                else:
                    # Multi-token log-likelihood calculation
                    full_text = f"{prompt}{opt_str}"
                    full_inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
                    full_outputs = self.model(**full_inputs)
                    full_logits = full_outputs.logits[0]

                    prompt_len = prompt_inputs["input_ids"].shape[1]
                    target_token_ids = full_inputs["input_ids"][0, prompt_len:]

                    log_prob = 0.0
                    for idx, target_id in enumerate(target_token_ids):
                        pos = prompt_len - 1 + idx
                        pos_logits = full_logits[pos]
                        pos_log_prob = F.log_softmax(pos_logits, dim=-1)[target_id].item()
                        log_prob += pos_log_prob

                option_log_likelihoods[option] = log_prob

        # Softmax over log likelihoods across all options
        log_probs_tensor = torch.tensor([option_log_likelihoods[opt] for opt in options])
        probs = F.softmax(log_probs_tensor, dim=0).tolist()

        probabilities = {opt: prob for opt, prob in zip(options, probs)}
        best_option = max(probabilities, key=probabilities.get)
        confidence = probabilities[best_option]

        return {
            "selected_option": best_option,
            "confidence": confidence,
            "probabilities": probabilities,
            "raw_log_likelihoods": option_log_likelihoods
        }

    def classify_noul(
        self,
        state: str,
        question: str,
        instructions: str = "Answer Yes or No."
    ) -> Dict[str, Any]:
        """
        Answers a binary (Yes/No) question on the state by evaluating token logits.
        """
        result = self.classify_choice(
            state=state,
            options=["Yes", "No"],
            instructions=f"{instructions} Question: {question}"
        )
        yes_prob = result["probabilities"].get("Yes", 0.0)
        return {
            "is_true": yes_prob >= 0.5,
            "probability": yes_prob,
            "details": result
        }

    def invoke(self, input_data: Dict[str, Any], config: Optional[Any] = None) -> Dict[str, Any]:
        """
        LangChain Runnable interface invocation.
        """
        state = input_data.get("state", "")
        options = input_data.get("options", None)
        question = input_data.get("question", None)
        instructions = input_data.get("instructions", "Classify the given context.")

        if options:
            return self.classify_choice(state=state, options=options, instructions=instructions)
        elif question:
            return self.classify_noul(state=state, question=question, instructions=instructions)
        else:
            raise ValueError("Input data must contain either 'options' or 'question'.")
