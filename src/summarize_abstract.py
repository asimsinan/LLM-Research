
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
def summarize():
    checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)
    input_text = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."

    summarizer = pipeline(
        task="summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=200,
        min_length=50
    )
    # Generate the summary
    output = summarizer(input_text)
    summary = output[0]['summary_text']


    bullet_points = summary.split(". ")
    for point in bullet_points:
        print(f"- {point}")
if __name__ == "__main__":
    summarize()