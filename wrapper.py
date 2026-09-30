from langchain_core.language_models.llms import LLM
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import torch

from semif import SemIfClassifier

# Custom LLM Class (Generative / System Two)
class LangLLM(LLM):
    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "LLM Wrapper for Langchain"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, chatbot=None) -> str:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        generation_args = {
            "max_new_tokens": 100,
            "return_full_text": False,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 50
        }
        output = pipe(prompt, **generation_args)
        response = output[0]['generated_text']
        return response

if __name__ == "__main__":
    print("=== Demo 1: Fast System One Decision Engine (SemIf Logit Reader) ===")
    semif_classifier = SemIfClassifier(model_name_or_path="openai-community/gpt2")

    # Example 1: Model/Intent Routing
    user_request = "I need immediate help connecting my account, it keeps crashing!"
    route_result = semif_classifier.invoke({
        "state": user_request,
        "options": ["TECHNICAL_SUPPORT", "BILLING", "GENERAL_FAQ"],
        "instructions": "Classify customer ticket intent."
    })
    print(f"User Request: {user_request}")
    print(f"Selected Intent Route: {route_result['selected_option']} (Confidence: {route_result['confidence']:.4f})")
    print(f"Option Probabilities: {route_result['probabilities']}\n")

    # Example 2: Binary Question (Noul)
    urgency_result = semif_classifier.invoke({
        "state": user_request,
        "question": "Is this issue time-sensitive or urgent?",
        "instructions": "Determine if the request conveys high urgency."
    })
    print(f"Urgent? {urgency_result['is_true']} (Probability: {urgency_result['probability']:.4f})\n")

    print("=== Demo 2: System Two Generative LLM (LangChain Custom Wrapper) ===")
    # Instantiate the LangChain LLM
    llm = LangLLM()

    # Perform LangChain operations
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="You are a helpful assistant. Answer the question: {question}"
    )
    memory = ConversationBufferMemory(llm=llm, max_token_limit=100)
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    # Query the model
    response = conversation.predict(input="Hi. Have you read the poem 'dulce et decorum est'. What's your take on it?")
    print(f"Conversation Response:\n{response}")
