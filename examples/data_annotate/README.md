# on-policy difficulty 어노테이션 (간단 버전)

`annotate_difficulty_vllm.py`는 parquet 데이터를 읽어서,
vLLM(OpenAI-compatible endpoint)로 여러 번 샘플링한 pass-rate 기반 난이도를 저장합니다.

## 실행 예시

```bash
python examples/data_annotate/annotate_difficulty_vllm.py \
  --input_parquet /data/train.parquet \
  --output_parquet /data/train_annotated.parquet \
  --vllm_base_url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

## 필수/핵심 옵션

- `--input_parquet` (여러 개 가능)
- `--output_parquet`
- `--vllm_base_url`
- `--model`
- `--temperature`, `--top_p`, `--max_tokens`
- `--num_samples_per_prompt`
- `--concurrency`, `--batch_size`
- `--overwrite`
- `--dry_run_n`

## 출력 컬럼

- `on_policy_difficulty` = `1 - pass_rate`
- `pass_rate`
- `difficulty_num_samples`
- `on_policy_difficulty_model`

