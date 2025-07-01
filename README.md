---
title: JarvisAgent for GAIA Benchmark
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

# 🚀 GAIA Solver Agent - Optimized & Production Ready

A highly optimized AI agent for the GAIA benchmark with robust error handling, parallel processing, and graceful API key management.

## ✨ Key Features

### 🚀 **Performance Optimizations**
- **⚡ Parallel Processing**: Process multiple questions concurrently using ThreadPoolExecutor
- **💾 Smart Caching**: File-based JSON cache to avoid reprocessing questions
- **🔄 Async Operations**: Non-blocking UI with real-time progress updates
- **📦 Batch Processing**: Questions processed in configurable batches for optimal performance

### 🛡️ **Robust Error Handling**
- **🔧 Graceful API Key Management**: Works with or without API keys
- **🔄 Smart Fallbacks**: Automatic fallback to free alternatives (DuckDuckGo vs Google Search)
- **🛡️ Error Recovery**: Individual question failures don't stop the entire process
- **📊 Comprehensive Logging**: Detailed status updates and error reporting

### 🧰 **Enhanced Tools**
- **🔍 Google Search** (with DuckDuckGo fallback)
- **📊 Math Solver** (SymPy-based calculations)
- **✂️ Text Preprocesser** (with enhanced reversal handling)
- **📖 Wikipedia Access** (title finder + content fetcher)
- **📁 File Analysis** (Gemini-powered document processing)
- **🎥 Video Analysis** (YouTube/video content analysis)
- **🧩 Riddle Solver** (pattern analysis for logic puzzles)
- **🌐 Web Page Fetcher** (HTML to markdown conversion)

## 🔧 Quick Start

### 1. **Installation**
```bash
git clone <your-repo>
cd GAIA-Solver-Agent
pip install -r requirements.txt
```

### 2. **Run the Agent**
```bash
python app.py
```

## 🔑 API Key Setup

### **Required for Full Functionality**

#### **Google/Gemini API (Recommended)**
```bash
# Get your key: https://makersuite.google.com/app/apikey
export GOOGLE_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here"  # Can be same as GOOGLE_API_KEY
```

#### **Google Custom Search (Optional)**
```bash
# Get search key: https://developers.google.com/custom-search/v1/introduction
# Create search engine: https://programmablesearchengine.google.com/
export GOOGLE_SEARCH_API_KEY="your_search_key"
export GOOGLE_SEARCH_ENGINE_ID="your_engine_id"
```

### **Graceful Fallbacks**

| Feature | With API Key | Without API Key |
|---------|-------------|-----------------|
| **Web Search** | Google Custom Search | DuckDuckGo (free) |
| **File Analysis** | Gemini-powered | Error message with setup guide |
| **Video Analysis** | Gemini-powered | Error message with setup guide |
| **Math/Text/Wikipedia** | ✅ Always available | ✅ Always available |

---