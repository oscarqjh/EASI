# .venv/bin/python scripts/run_easi_mini.py \
#   --backend vllm \
#   --model OpenGVLab/InternVL3-8B \
#   --num-parallel 2 \
#   --llm-instances 1 \
#   --llm-gpus 0,1 \
#   --llm-kwargs '{"tensor_parallel_size": 2, "trust_remote_code": true, "chat_template": "easi/llm/templates/internvl3.jinja"}' \
#   --verbosity TRACE

.venv/bin/python scripts/run_easi_mini.py \
      --backend custom --model internvl3 \
      --aggregate-only \
      --output-dir logs/easi_mini/20260317_2238_internvl3

# .venv/bin/python scripts/run_easi_mini.py \
#   --backend custom \
#   --model internvl3 \
#   --num-parallel 1 \
#   --llm-instances 1 \
#   --llm-gpus 0,1 \
#   --llm-kwargs '{"model_path": "sensenova/SenseNova-SI-1.3-InternVL3-8B"}' \
#   --verbosity TRACE

# .venv/bin/python scripts/run_easi_mini.py \
#   --backend custom \
#   --model internvl3 \
#   --num-parallel 1 \
#   --llm-instances 1 \
#   --llm-gpus 0,1 \
#   --llm-kwargs '{"model_path": "OpenGVLab/InternVL3-8B"}' \
#   --verbosity TRACE

# .venv/bin/python scripts/run_easi_mini.py \
#   --backend vllm --model OpenGVLab/InternVL3-8B \
#   --tasks ebalfred_spatial,ebhabitat_spatial_relationship,ebmanipulation_complex,ebmanipulation_spatial,ebmanipulation_visual,ebnavigation_common_sense,ebnavigation_complex_instruction,ebnavigation_visual_appearance,vlnce_r2r_val_unseen \
#   --output-dir logs/easi_mini/20260317_0907_InternVL3-8B \
#   --llm-kwargs '{"tensor_parallel_size": 1, "trust_remote_code": true, "chat_template_content_format": "openai"}' \
#   --num-parallel 2 \
#   --llm-instances 1 \
#   --llm-gpus 0,1 \
#   --verbosity TRACE

# .venv/bin/python scripts/run_easi_mini.py \
#   --backend openai --model gpt-5.2-2025-12-11 \
#   --num-parallel 8 --llm-instances 2 \
#   --llm-gpus 0,1,2,3 --sim-gpus 4,5 \
#   --llm-kwargs '{"tensor_parallel_size": 2}'

  # .venv/bin/python scripts/run_easi_mini.py \
  #     --backend openai --model gpt-4o \
  #     --aggregate-only \
  #     --output-dir logs/easi_mini/<your_run_dir>
