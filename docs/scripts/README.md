# Script Documentation

This directory contains detailed documentation for all MindCaptioning Python scripts.

## Main Analysis Scripts

### Encoding & Decoding
- [mcap_encoding_analysis.md](./mcap_encoding_analysis.md) - Encoding analysis from brain to features
- [mcap_decoding_analysis.md](./mcap_decoding_analysis.md) - Decoding analysis from brain to features
- [mcap_analysis.md](./mcap_analysis.md) - Text generation from decoded features
- [mcap_evaluation.md](./mcap_evaluation.md) - Evaluation of generated text

### Visualization
- [mcap_summary_encoding.md](./mcap_summary_encoding.md) - Encoding results visualization
- [mcap_summary_decoding.md](./mcap_summary_decoding.md) - Decoding results visualization

### Demo
- [mcap_demo.md](./mcap_demo.md) - Interactive Jupyter notebook demo

## Utility Modules

### Configuration & Core
- [mcap_config.md](./mcap_config.md) - Configuration management
- [mcap_utils.md](./mcap_utils.md) - General utility functions
- [mcap_utils_demo.md](./mcap_utils_demo.md) - Demo-specific utilities

### Analysis Modules
- [mcap_encoding.md](./mcap_encoding.md) - Encoding analysis functions
- [mcap_decoding.md](./mcap_decoding.md) - Decoding analysis functions
- [thutil4.md](./thutil4.md) - Helper functions

## Quick Reference

### Complete Analysis Pipeline

```bash
# 1. Encoding Analysis (~1 week on multi-CPU)
python mcap_encoding_analysis.py

# 2. Decoding Analysis (~1-2 weeks on multi-CPU)
python mcap_decoding_analysis.py

# 3. Text Generation (~1 week on multi-GPU)
python mcap_analysis.py

# 4. Evaluation
python mcap_evaluation.py

# 5. Visualization
python mcap_summary_encoding.py
python mcap_summary_decoding.py
```

### Quick Demo

```bash
# Interactive demo (no heavy computation needed)
jupyter notebook mcap_demo.ipynb
```

## Script Dependencies

```
mcap_encoding_analysis.py
├── util/mcap_config.py
├── util/mcap_encoding.py
├── util/mcap_utils.py
└── util/thutil4.py

mcap_decoding_analysis.py
├── util/mcap_config.py
├── util/mcap_decoding.py
├── util/mcap_utils.py
└── util/thutil4.py
    ↓ (uses encoding results)

mcap_analysis.py
├── util/mcap_utils_demo.py
└── util/thutil4.py
    ↓ (uses decoding results)

mcap_evaluation.py
├── util/mcap_utils_demo.py
└── util/thutil4.py
    ↓ (uses text generation results)

mcap_summary_*.py
└── (uses analysis results)
```

## Documentation Structure

Each script documentation includes:

1. **Overview**: What the script does
2. **Purpose**: Why it's needed
3. **Usage**: How to run it
4. **Parameters**: Configuration options
5. **Inputs**: Required data
6. **Outputs**: Generated results
7. **Algorithm**: Technical details
8. **Performance**: Time/memory requirements
9. **Troubleshooting**: Common issues
10. **Examples**: Usage examples

## Getting Help

- **General**: See [../OVERVIEW.md](../OVERVIEW.md)
- **Installation**: See [../INSTALLATION.md](../INSTALLATION.md)
- **Usage Guide**: See [../USAGE.md](../USAGE.md)
- **API Reference**: See [../API.md](../API.md)
