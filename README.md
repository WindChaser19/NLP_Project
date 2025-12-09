# NLP_Project
NLP (6.461) project code

Code is in `dual_hirope_eval_submit.py`. Testing GovReport with fixed blocking is implemented by making sure in main() that  test_govreport(model, tokenizer, args, dual="block") is uncommented. Testing GovReport with semantic positions is implemented by uncommenting test_govreport(model, tokenizer, args, dual="dual") in main(). Similarly narrativeQA is ran by calling test_narrativeqa(model, tokenizer, args) in main().   

Specify the positional scheme with --rope as "hirope", "sumrope" or "sumhirope" , and the model size as --model with value "Qwen/Qwen3-1.7B" ""Qwen/Qwen3-0.6B" or "Qwen/Qwen3-4B" . We ran with --n as 100 (number of samples). 
