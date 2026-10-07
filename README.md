# R2AI Stage 1: truy hồi văn bản pháp luật tiếng Việt

Mã nguồn này triển khai phương pháp cơ bản được mô tả trong tài liệu kỹ thuật: phân đoạn theo cấu trúc pháp luật, truy hồi bằng BM25 và dense embedding, hợp nhất bằng RRF, xếp hạng lại bằng cross-encoder, rồi dùng MaxP để giữ một đoạn đại diện cho mỗi điều. Kết quả cuối gồm tối đa ba điều cho mỗi câu hỏi.

Hai mô hình được sử dụng là [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding) và [AITeamVN/Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker). Cấu hình mặc định cố định phiên bản mô hình theo commit để kết quả không thay đổi ngoài ý muốn.

Kho văn bản dùng để lập chỉ mục và nhãn kiểm thử không được phát hành kèm mã nguồn. Lệnh `build` nhận tệp JSONL đã tách đến cấp điều. Mỗi dòng cần có `article_id`, `doc_id`, `article_number`, `article_reference` và `text`; `text` chứa toàn bộ nội dung điều, gồm tiêu đề và phần dẫn nhập nếu có. Lệnh `run` nhận một mảng JSON gồm các trường `id` và `question`.

Môi trường yêu cầu Python 3.10 trở lên và GPU hỗ trợ CUDA. Cài đặt các phụ thuộc bằng `pip install -e .`.

## Dựng chỉ mục

```bash
python -m baseline.cli build \
  --corpus data/articles.jsonl \
  --index artifacts/index \
  --config config.json
```

## Chạy truy hồi

```bash
python -m baseline.cli run \
  --questions data/questions.json \
  --index artifacts/index \
  --output outputs/results.json \
  --config config.json
```

`results.json` giữ nguyên `id` và `question`, để trống `answer`, đồng thời ghi danh sách `relevant_docs` và `relevant_articles` theo thứ hạng truy hồi.

## Dữ liệu tham khảo

Nếu chưa có dữ liệu đầu vào, có thể tham khảo [gói dữ liệu đã xử lý](https://drive.google.com/drive/folders/1NvkAEpOjOqHNqF9kC-FvrOs5fjIiscTj) do **Hung&Fong**, đội đạt giải Nhì R2AI Stage 1, chia sẻ. Kết quả cuộc thi được công bố trong [repository R2AI Stage 1](https://github.com/aigurutinix/r2ai-stage-1/blob/main/README.en.md); quy trình xử lý của đội được trình bày trong [thư mục Hung&Fong](https://github.com/aigurutinix/r2ai-stage-1/tree/main/hung%26phong). Đây là một lựa chọn đầu vào bổ sung, tách biệt với phần mô tả thiết lập thực nghiệm.

Gói dữ liệu sử dụng định dạng riêng. Trước khi chạy lệnh `build`, chuyển các điều luật cần dùng về schema JSONL gồm `article_id`, `doc_id`, `article_number`, `article_reference` và `text` như mô tả ở trên. Mã nguồn không phụ thuộc vào cấu trúc tệp hay chỉ mục có sẵn trong gói dữ liệu này.
