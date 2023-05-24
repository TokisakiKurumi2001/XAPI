import evaluate
metrics = evaluate.load('metrics/paraid.py')
predictions = [0, 1, 1]
references = [1, 1, 1]
result = metrics.compute(predictions=predictions, references=references)
print(result)