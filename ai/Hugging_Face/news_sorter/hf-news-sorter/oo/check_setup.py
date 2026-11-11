"""Smoke test: run a tiny sentiment-analysis pipeline to check the installation."""
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("Hugging Face runs fine on my laptop!"))
