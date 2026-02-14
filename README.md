# Master Thesis: CUDA Code Generation Method Based on Dataflow Analysis

This repository contains the LaTeX source code for the Master's Thesis of **Chengxiang Qi** at the **University of Chinese Academy of Sciences (UCAS)**.

**Title:** Based on Dataflow Analysis of CUDA Code Generation Method (基于数据流分析的CUDA代码生成方法)  
**Author:** Chengxiang Qi (齐呈祥)  
**Advisor:** Prof. Yongjun Xu (徐勇军)  
**Institute:** Hangzhou Institute for Advanced Study, UCAS (国科大杭州高等研究院)

## 📂 Project Structure

The project is organized as follows to keep the root directory clean and manageable:

```
.
├── Thesis.tex              # Main entry point for the LaTeX document
├── contents/               # Thesis chapters and content
│   ├── abstract_cn.tex     # Chinese Abstract
│   ├── abstract_en.tex     # English Abstract
│   ├── chap0*.tex          # Chapters (Introduction, Related Work, etc.)
│   └── ...
├── setup/                  # Configuration and style definitions
│   ├── info.tex            # Thesis metadata (Title, Author, etc.)
│   ├── packages.tex        # Package imports and global settings
│   ├── define-language.tex # Code highlighting definitions
│   ├── style/              # Custom style files (.sty, .cls, .cfg)
│   └── bib-styles/         # Bibliography style files (.bst, .bbx)
├── bib/                    # Bibliography data
│   └── ref.bib             # BibTeX database
├── figures/                # Figures and images used in the thesis
├── scripts/                # Compilation scripts
│   ├── artratex.sh         # Build script for Linux/macOS
│   └── artratex.bat        # Build script for Windows
└── eval_repos/             # Submodules for evaluation code (e.g., cutlass, flash-attention)
```

## 🚀 Prerequisites

To compile this project, you need a standard LaTeX distribution installed on your system:

- **TeX Live** (Recommended for Linux/macOS/Windows)
- **MiKTeX** (Alternative for Windows)
- **MacTeX** (macOS)

Ensure that `xelatex` or `pdflatex` and `bibtex` (or `biber`) are available in your system's PATH.

## 🛠 Compilation

We provide automated scripts to build the thesis PDF easily.

### Linux / macOS

Run the shell script from the project root:

```bash
./scripts/artratex.sh
```

By default, this uses `xelatex` and `bibtex`. You can specify the engine if needed (see script usage).

### Windows

Run the batch script from the project root:

```cmd
.\scripts\artratex.bat
```

### Manual Compilation

If you prefer to compile manually, the standard sequence is:

```bash
xelatex Thesis
bibtex Thesis
xelatex Thesis
xelatex Thesis
```

## 📝 Writing

- **Metadata**: Update title, author, and date in `setup/info.tex`.
- **Content**: Edit the chapters in the `contents/` directory.
- **References**: Add BibTeX entries to `bib/ref.bib`.
- **Styles**: Modify `setup/packages.tex` for package imports or `setup/style/` for deep customization.

## 📄 License & Credits

This project is based on the [ucasproposal](https://github.com/mohuangrui/ucasthesis) template by Huangrui Mo.

The content of the thesis is determining. The template code is subject to the license of the original `ucasthesis` project.
