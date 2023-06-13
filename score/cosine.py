import numpy as np
import time

def cosine_score(trials, index_mapping, eval_vectors):
    labels = []
    scores = []
    verification_times = []
    for item in trials:
        enroll_vector = eval_vectors[index_mapping[item[1]]]
        test_vector = eval_vectors[index_mapping[item[2]]]
        start = time.time()
        score = enroll_vector.dot(test_vector.T)
        denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
        score = score/denom
        verification_times.append(time.time() - start)
        labels.append(int(item[0]))
        scores.append(score)
    print('Average verification time: ', sum(verification_times)/len(verification_times))
    print('Max verification time: ', max(verification_times))
    print('Min verification time: ', min(verification_times))
    return labels, scores

