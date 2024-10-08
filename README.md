## CodeGLEU

While CodeBLEU performs well on sentence-to-sentence comparisons or comparisons between small methods (Ren et al, 2020), the score quickly degenerates when used to compare large methods or files which were only partially modified.

This occurs due to an deficit with the original BLEU in regards to text modification or synthesis tasks. Since BLEU considers n-gram precision, simply repeating the source text leads to a high score in monolingual tasks (Napoles et al, 2015). The smaller the changes made by the references, the higher the score of the source sentence. 

In the context of CodeBLEU and code-modification, this means models are often rewarded not just for making appropriate changes, but also for making as few changes as possible, severely polluting the metric.

This deficit was successfully resolved for textual tasks with GLEU (Napoles et al, 2015), which introduces a penalty for n-grams that should have been modified but were not, bringing the metric into alignment with human evaluations even for complex grammatical error correction tasks. 

A similarly improved metric for the evaluation of code synthesis does not yet, however, exist. As such, i am introducing the CodeGLEU metric, which aims to extend CodeBLEU in the same way that GLEU extends BLEU.

## TODO
- [X] Rewrite ngram_match_score to use GLEU formula    
- [X] Rewrite weighted ngram_match_score to use GLEU formula    
- [X] Rewrite syntax_match_score to use GLEU formula    
- [X] Rewrite dataflow_match_score to use GLEU formula    
- [X] Write Tests    
- [X] Evaluate on suitable dataset    
- [X] ensure codegleu without penalty == codebleu
- [X] 4-grams enough?: Minor improvement (+0.02%) in correlation at 10 ngrams vs 4
- [X] check n-gram correlation - all about equivalent, at 0.14 or so
- [X] check correlation with num lines changes / num lines total - bleu 0.8, codebleu 0.65, codegleu 0.5 - still bad
- [X] limit to only changes/snippeting?

## Related Works / Citations
Papineni et al, 2002: [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf)    
Ren et al, 2020: [CodeBLEU: a Method for Automatic Evaluation of Code Synthesis](https://arxiv.org/pdf/2009.10297)    
Napoles et al, 2015: [Ground Truth for Grammatical Error Correction Metrics](https://aclanthology.org/P15-2097.pdf)    
Napoles et al, 2016: [GLEU Without Tuning](https://arxiv.org/pdf/1605.02592)    