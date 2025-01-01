from collections import Counter
from datasets import load_dataset
from encodzall import ByteLevelTokenizer
import numpy as np
from scipy import stats
import tqdm
import more_itertools
import difflib

# Load the dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[0:10000]")

tokenizer = ByteLevelTokenizer()
c = Counter()

for row in tqdm.tqdm(dataset):
    text = row["text"]
    byte_seq = tokenizer.tokenize(text, char_len = 819200000, truncate_len = 1024000000, noise_prob=0, mask_prob=0, return_byte_seq=True)
    lengths = [len(x) for x in byte_seq]
    raw = bytearray(more_itertools.flatten(byte_seq)).decode()
    try:
        assert raw == text
    except:
        differ = difflib.Differ()
        diff = differ.compare(text.splitlines(), raw.splitlines())
        
        # Join the diff output into a single string
        diff_output = "\n".join(diff)
        print(f"TEXT: {text}\n\nRAW: {raw}\n\n")
        print(diff_output)
        raise
    c.update(lengths)



expanded_data = []
for word_size, count in c.items():
    expanded_data.extend([word_size] * count)

# Convert to numpy array
expanded_data = np.array(expanded_data)

# Calculate the interquartile range (IQR)
q1 = np.percentile(expanded_data, 2)
q2 = np.percentile(expanded_data, 50)
q3 = np.percentile(expanded_data, 98)
iqr = q3 - q1

print(q2, iqr)
print(sum(c.values()))
print(c)

