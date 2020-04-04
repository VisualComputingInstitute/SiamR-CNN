import random


def subsample_nns(query_seq, nns, names, n_seqs_to_sample, remove_query=True):
    random.shuffle(nns)

    nn_names = [names[n] for n in nns]
    nn_seqs = [x.split("/")[-2] for x in nn_names]

    seq_to_nns = {}
    for nn, nn_seq in zip(nns, nn_seqs):
        if nn_seq not in seq_to_nns:
            seq_to_nns[nn_seq] = []
        seq_to_nns[nn_seq].append(nn)

    #seq_to_n = sorted([(k, len(v)) for k, v in seq_to_nns.items()], key=lambda x: x[1], reverse=True)
    #n_total = sum(x[1] for x in seq_to_n)
    #for seq, n_in_seqs in seq_to_n:
    #    pct = n_in_seqs * 100 / n_total
    #    if pct > 1.0:
    #        print(seq, pct, "%")
    #print("n_seqs in nns", len(seq_to_nns))

    sampled_nns = []
    sample_seqs = set(seq_to_nns.keys())
    if remove_query:
        sample_seqs.remove(query_seq)
    sample_seqs = list(sample_seqs)
    random.shuffle(sample_seqs)
    sample_seqs = sample_seqs[:n_seqs_to_sample]

    # get 1 per sequence
    for seq in sample_seqs:
        seq_nns = seq_to_nns[seq]
        nn = random.choice(seq_nns)
        sampled_nns.append(nn)
    return sampled_nns


def subsample_nns_old(name, nns, names, n_seqs_to_sample):
    random.shuffle(nns)

    nn_names = [names[n] for n in nns]
    nn_seqs = [x.split("/")[-2] for x in nn_names]
    seq = name.split("/")[-2]

    sampled_nns = []
    sampled_seqs = set()
    sampled_seqs.add(seq)
    # get 1 per sequence
    for nn, seq in zip(nns, nn_seqs):
        if seq not in sampled_seqs:
            sampled_seqs.add(seq)
            sampled_nns.append(nn)
    sampled_nns = sampled_nns[:n_seqs_to_sample]
    return sampled_nns
