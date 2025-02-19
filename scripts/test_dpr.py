import torch
from torch import nn
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
from typing import List, Tuple

class DPRRetriever(nn.Module):
    def __init__(self, question_encoder, question_tokenizer, context_encoder, context_tokenizer):
        super().__init__()
        self.question_encoder = question_encoder
        self.context_encoder = context_encoder
        self.question_tokenizer = question_tokenizer
        self.context_tokenizer = context_tokenizer

    def encode_passages(self, passages: List[str]) -> torch.Tensor:
        """
        Encode a list of passages into dense vectors.
        """
        input_ids = self.context_tokenizer(passages, padding=True, return_tensors='pt', max_length=128, truncation=True).input_ids.cuda()
        passage_embeds = self.context_encoder(input_ids)[0]
        return passage_embeds

    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a query into a dense vector.
        """
        input_ids = self.question_tokenizer(query, padding=True, return_tensors='pt', max_length=128, truncation=True).input_ids.cuda()
        query_embed = self.question_encoder(input_ids)[0]
        
        return query_embed

    def retrieve_top_k(self, query: str, passages: List[str], k: int = 5) -> List[Tuple[int, float]]:
        """
        Retrieve the top-k most relevant passages for a given query.
        """
        query_embed = self.encode_query(query)
        passage_embeds = self.encode_passages(passages)

        # Compute cosine similarity between query and passage embeddings
        similarities = torch.matmul(query_embed.unsqueeze(0), passage_embeds.T).squeeze(0)

        # Get the indices of the top-k most similar passages
        top_k_indices = torch.topk(similarities, k, largest=True, sorted=True).indices.tolist()[0]  # [0] transform [[]] into []
        # Return the top-k indices and their similarity scores
        return [(idx, float(similarities[0][idx])) for idx in top_k_indices]
        


if __name__ == '__main__':
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    print(question_model.device)

    retriever = DPRRetriever(question_model, question_tokenizer, context_model, context_tokenizer)
    passages = ["This is the first passage.", "This is the second passage.", "This is the third passage.", "This is the fourth passage.","This is the fifth passage.","This is the last passage."]

    # 检索与查询最相关的前 5 个文本段落
    query = "What is the topic of the first passage?"
    top_k_indices_and_scores = retriever.retrieve_top_k(query, passages, k=5)
    top_k_results = [passages[item] for item, score in top_k_indices_and_scores]
    print(top_k_results)