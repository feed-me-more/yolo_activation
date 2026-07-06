import numpy as np

NUM_PACKETS = 24      # number of packets per frame
NUM_PATTERNS = 3000   # how many patterns to generate

rng = np.random.default_rng()   #Use seed = 42 for reproducibility

patterns = []
seen = set()
out_path = "interleave_patterns.npy"

blk_int = [0,12,1,13,2,14,3,15,4,16,5,17,6,18,7,19,8,20,9,21,10,22,11,23]    # block interleave
even_odd = [0,2,4,6,8,10,12,14,16,18,20,22,1,3,5,7,9,11,13,15,17,19,21,23]    # even-odd
reverse = [23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]    # reverse


patterns.append(blk_int)
seen.add(tuple(blk_int))

patterns.append(even_odd)
seen.add(tuple(even_odd))

patterns.append(reverse)
seen.add(tuple(reverse))

seen.add(tuple([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]))

while len(patterns) < NUM_PATTERNS:
    perm = rng.permutation(NUM_PACKETS).tolist()
    # print(f"Generated pattern: {perm}")

    if tuple(perm) not in seen:
        seen.add(tuple(perm))
        patterns.append(list(perm))

arr = np.array(patterns, dtype=np.int64)

assert arr.shape == (NUM_PATTERNS, NUM_PACKETS)
assert len(set(map(tuple, arr.tolist()))) == NUM_PATTERNS

np.save(out_path, arr)
print("Saved:", out_path)
print("Shape:", arr.shape)
# print("First 3 rows:\n", arr[:3])
print("Unique rows:", len(set(map(tuple, arr.tolist()))))

