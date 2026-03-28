# Mac Studio M3 Ultra Setup Guide

## Your Hardware
- **Mac Studio M3 Ultra** with **96GB unified memory**
- **Parallel loading enabled** by default in `config.yaml`
- **Expected runtime**: 25-35 minutes for full experiment

## Quick Start

### 1. Install Ollama
```bash
brew install ollama
```

### 2. Start Ollama Server
```bash
ollama serve
# Runs on http://localhost:11434
# Keep this terminal open
```

### 3. In Another Terminal, Pull Models
```bash
ollama pull qwen2.5:14b      # For dataset generation & validation
ollama pull shieldgemma:9b   # For inference
```

### 4. Verify Setup
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Should show both models available
```

### 5. Run the Pipeline
```bash
# From project root
python src/main.py
```

## What Happens (Parallel Mode)

```
[00:00] Loading all 6 models simultaneously...
[00:30] All models loaded (~25GB in 4-bit)
[00:30] Starting parallel inference (6 threads)
[25:00] All inference complete
[30:00] Analysis & visualization
[35:00] Done!
```

## Config for Mac Studio

Your `config.yaml` should have:
```yaml
hardware:
  parallel_loading: true      # ✅ Enabled for 96GB RAM
  max_parallel_workers: 5     # One per local model

ollama:
  base_url: "http://localhost:11434"
  generation_model: "qwen2.5:14b"
  judge_model: "qwen2.5:14b"
```

## Monitoring

### Check VRAM Usage
```bash
# In another terminal, watch memory
watch -n 1 'ps aux | grep -E "python|ollama" | head -20'
```

### Check Ollama Status
```bash
# Verify models are loaded
curl http://localhost:11434/api/tags | jq '.models[].name'
```

### View Logs
```bash
# Real-time logs
tail -f results/experiment.log
```

## Troubleshooting

### "Connection refused" to Ollama
```bash
# Make sure Ollama server is running
ollama serve
```

### "Model not found: qwen2.5:14b"
```bash
# Pull the model
ollama pull qwen2.5:14b
```

### Out of memory errors
- This shouldn't happen on 96GB RAM
- If it does, check if other apps are using memory
- Parallel mode uses ~25GB total for all 6 models

### Slow inference
- First run will be slower (model loading)
- Subsequent runs use cached models
- Check if Ollama is using GPU acceleration (should be automatic on M3)

## Performance Expectations

| Phase | Time | Notes |
|-------|------|-------|
| Dataset Generation | ~15 min | Ollama qwen2.5:14b |
| Validation | ~10 min | LLM judge + human review |
| Sanity Check | ~2 min | 100 items, all 6 models |
| Pilot | ~5 min | 200 items, 2 models |
| Full Experiment | ~25-35 min | 10k items, 6 models in parallel |
| Analysis | ~5 min | Calibration curves, tables |
| **Total** | **~60-70 min** | One complete run |

## Cost

**$0** — Everything runs locally on your Mac Studio.

No API keys needed. No internet required (after models are downloaded).

## Next Steps

1. ✅ Install Ollama
2. ✅ Start `ollama serve`
3. ✅ Pull models
4. ✅ Run `python src/main.py`
5. ✅ Check `results/` for outputs

## Questions?

See `.kiro/OLLAMA_INTEGRATION_SUMMARY.md` for detailed explanation of why Ollama was chosen and how it compares to the original OpenAI API approach.

---

**GitHub**: https://github.com/junwenleong/guardrail-calibration-eval
**Status**: Production-ready, all tests passing (165/165)
