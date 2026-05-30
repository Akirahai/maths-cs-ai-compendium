# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

An open, intuition-first textbook covering mathematics, CS, and AI — 20 chapters of Markdown content, each with 5–8 sections, plus SVG diagrams in `images/` and an MCP server in `mcp/`. Published via MkDocs Material to GitHub Pages.

## Site — MkDocs

```bash
pip install mkdocs-material
mkdocs serve          # local preview at http://127.0.0.1:8000
mkdocs build          # build static site into site/
```

The site root is the repo root (`docs_dir: .` in `mkdocs.yml`). Every Markdown file listed under `nav:` in `mkdocs.yml` is a page. MathJax is loaded via `javascripts/mathjax.js`; inline math uses `$...$` and block math uses `$$...$$`.

## MCP Server

The `mcp/` directory is a TypeScript MCP server that exposes the compendium as a knowledge base to AI assistants.

```bash
cd mcp
npm install           # first-time setup
npm start             # run the server (uses tsx, no compile step needed)
```

The server reads the repo root via `process.env.COMPENDIUM_ROOT` (defaults to `../../` relative to `src/index.ts`). Tools exposed: `list_topics`, `read_section`, `search`, `recommend`, `get_examples`.

To build a compiled JS bundle:
```bash
cd mcp
npx tsc               # outputs to mcp/dist/
```

## Content Structure

Chapters follow strict naming conventions the MCP server relies on for discovery:

- Chapter directories: `chapter NN: <name>/` (two-digit zero-padded number, colon, lowercase name)
- Section files: `NN. <name>.md` (two-digit number, period, lowercase name)

The `llms.txt` file in the root is a structured index consumed by the MCP `recommend` tool — update it when adding or renaming sections.

## Adding or Editing Content

1. Write content in the appropriate chapter directory following the naming convention above.
2. Add the new file to the `nav:` section of `mkdocs.yml` to make it appear in the site.
3. Add a description line to `llms.txt` under the correct chapter heading so the MCP `recommend` tool can surface it.
4. Place any new SVG diagrams in `images/` and reference them with a relative path from the section file.

## Content Style

Sections are intuition-first: explain the *why* before the *what*, give real-world context, and include code examples for concrete concepts. Math notation uses LaTeX via MathJax. Diagrams are SVGs kept in `images/`.
