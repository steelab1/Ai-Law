import torch
import logging
from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)

class BGEReranker:
    def __init__(self, model_name: str, use_fp16: bool = True):
        """
        Khởi tạo BGEReranker với model đã tải.
        Auto-detect GPU và sử dụng nếu có.
        """
        # Auto-detect device
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"BGEReranker using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("BGEReranker using CPU (no GPU available)")

        self.model = FlagReranker(model_name, use_fp16=use_fp16, device=device)

    def calculate_scores(self, pairs: list[list[str]]) -> list[float]:
        """
        Tính toán scores cho các cặp query-document.
        pair là định dạng: [[query, document]]
        """
        scores = self.model.compute_score(pairs, normalize=True)
        return scores

    def rerank(self, query: str, documents: list[str], topk: int = 10) -> list[str]:
        """
        Rerank danh sách documents theo scores.
        """
        pairs = [[query, doc] for doc in documents]
        list_scores = self.calculate_scores(pairs)

        # Kết hợp documents và scores
        doc_scores = list(zip(documents, list_scores))

        # Sắp xếp theo scores giảm dần
        sorted_doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        # Lấy top-k documents
        top_k_documents = [doc for doc, score in sorted_doc_scores[:topk]]
        return top_k_documents


if __name__ == "__main__":
    
    model_name = "/home/ivirse/ivirse_all_data/namnt/soict/checkpoint/rerank/bge_v2_part2/checkpoint-225000"
    reranker = BGEReranker(model_name)
    
    query = "Xe máy không đội mũ bảo hiểm bị phạt bao nhiêu tiền?"
    list_doc = [
        "Xe ô tô sẽ bị phạt 5000000 đ nếu đi sai làn đường.",
        "Nếu trong TH xe máy không đội mũ bảo hiểm sẽ bị phạt 200.000 đ.",
        "Nếu đi ngược chiều thì sẽ bị phạt nhiều tiền kể cả xe máy và ô tô."
    ]
    
    output = reranker.rerank(query, list_doc)
    
    print(output)
    
    
    

    