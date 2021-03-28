import torch
import argparse
from tqdm import tqdm
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_collection(collection_path):
    collection = {}
    with open(collection_path, 'r') as f:
        for line in tqdm(f, desc="Loading collection...."):
            docid, text = line.strip().split("\t")
            collection[docid] = text
    return collection


def load_queries(query_path):
    query = {}
    with open(query_path, 'r') as f:
        for line in tqdm(f, desc="Loading query...."):
            qid, text = line.strip().split("\t")
            query[qid] = text
    return query


def load_run(run_path, run_type='msmarco'):
    run = {}
    with open(run_path, 'r') as f:
        for line in tqdm(f, desc="Loading run file...."):
            if run_type == 'msmarco':
                qid, docid, score = line.strip().split("\t")
            elif run_type == 'trec':
                qid, _, docid, rank, score, _ = line.strip().split(" ")
            if qid not in run.keys():
                run[qid] = []
            run[qid].append(docid)
    return run


def main(args):
    collection_path = args.collection_path
    query_path = args.query_path
    run_path = args.run_path
    output_path = args.output_path
    rerank_cut = int(args.rerank_cut)
    batch_size = int(args.batch_size)
    run_type = args.run_type

    collection = load_collection(collection_path)
    queries = load_queries(query_path)
    run = load_run(run_path, run_type=run_type)

    tokenizer = T5Tokenizer.from_pretrained('t5-base', cache_dir=".cache")
    config = T5Config.from_pretrained('t5-base', cache_dir=".cache")
    model = T5ForConditionalGeneration.from_pretrained(
        't5-base-tf/model.ckpt-1004000', from_tf=True, config=config)
    model.to(DEVICE)


    for qid in tqdm(run.keys(), desc="Ranking queries...."):
        query = queries[qid]

        # split batch of documents in top 1000
        docids = run[qid]
        num_docs = min(rerank_cut, len(docids))  # rerank top k
        numIter = num_docs // batch_size + 1

        total_scores = []
        for i in range(numIter):
            start = i * batch_size
            end = (i + 1) * batch_size
            if end > num_docs:
                end = num_docs
                if start == end:
                    continue

            batch_passages = []
            for docid in docids[start:end]:
                batch_passages.append(collection[docid])

            inputs = tokenizer(batch_passages, return_tensors='pt', padding=True).to(DEVICE)
            labels = tokenizer([query] * (end-start), return_tensors='pt', padding=True).input_ids.to(DEVICE)

            with torch.no_grad():
                logits = model(**inputs, labels=labels, return_dict=True).logits

                distributions = torch.softmax(logits, dim=-1)  # shape[batch_size, decoder_dim, num_tokens]
                decoder_input_ids = labels.unsqueeze(-1)  # shape[batch_size, decoder_dim, 1]
                batch_probs = torch.gather(distributions, 2, decoder_input_ids).squeeze(-1)  # shape[batch_size, decoder_dim]
                masked_log_probs = torch.log10(batch_probs)  # shape[batch_size, decoder_dim]
                scores = torch.sum(masked_log_probs, 1)  # shape[batch_size]
                total_scores.append(scores)

        total_scores = torch.cat(total_scores).cpu().numpy()
        # rerank documents
        zipped_lists = zip(total_scores, docids)
        sorted_pairs = sorted(zipped_lists, reverse=True)

        # write run file
        lines = []
        for i in range(num_docs):
            score, docid = sorted_pairs[i]
            if run_type == 'msmarco':
                lines.append(qid + "\t" + docid + "\t" + str(i + 1) + "\n")
            if run_type == 'trec':
                lines.append(
                    qid + " " + "Q0" + " " + docid + " " + str(i + 1) + " " + str(score) + " " + "DeepQLM_t5" + "\n")

        with open(output_path, "a+") as f:
            f.writelines(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_path', required=False, default='data/collection.tsv', help='Path to MS MARCO passage collection.')
    parser.add_argument('--query_path', required=False, default='data/DL2019-queries.tsv', help='Path to query file.')
    parser.add_argument('--run_path', required=False, default='data/DL2019_bm25_default.res', help='Path to run file that needs to be reranked.')
    parser.add_argument('--run_type', required=False, default='trec', help='trec or msmarco')
    parser.add_argument('--rerank_cut', required=False, default=1000, help='Cut off for rerank postion')
    parser.add_argument('--batch_size', required=False, default=64, help='batch size')
    parser.add_argument('--output_path', required=False, default='data/DL2019_bm25_DeepQLM_t5_rerank.res', help='Run file output path.')
    args = parser.parse_args()

    main(args)
