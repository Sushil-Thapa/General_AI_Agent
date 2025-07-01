#!/usr/bin/env python3
"""
Configuration and API key management for GAIA Solver Agent
Handles missing API keys gracefully and provides user guidance
"""

import os
import sys
from typing import Dict, List, Optional

# Required API keys and their purposes
API_KEYS_INFO = {
    "GOOGLE_API_KEY": {
        "purpose": "Google Gemini AI for file analysis and video processing",
        "required_for": ["FileAttachmentQueryTool", "GeminiVideoQA", "Primary LLM"],
        "fallback": "Use DuckDuckGo search and text-only processing",
        "how_to_get": "https://makersuite.google.com/app/apikey"
    },
    "GEMINI_API_KEY": {
        "purpose": "Alternative Gemini API key (can be same as GOOGLE_API_KEY)",
        "required_for": ["LiteLLM model configuration"],
        "fallback": "Use GOOGLE_API_KEY if available",
        "how_to_get": "https://makersuite.google.com/app/apikey"
    },
    "GOOGLE_SEARCH_API_KEY": {
        "purpose": "Google Custom Search API for web searches",
        "required_for": ["GoogleSearchTool"],
        "fallback": "Use DuckDuckGo search (free but less comprehensive)",
        "how_to_get": "https://developers.google.com/custom-search/v1/introduction"
    },
    "GOOGLE_SEARCH_ENGINE_ID": {
        "purpose": "Google Custom Search Engine ID",
        "required_for": ["GoogleSearchTool"],
        "fallback": "Use DuckDuckGo search",
        "how_to_get": "https://programmablesearchengine.google.com/"
    }
}

# Optional environment variables
OPTIONAL_ENV_VARS = {
    "SPACE_ID": "Hugging Face Space ID (auto-detected in HF Spaces)",
    "SPACE_HOST": "Hugging Face Space host (auto-detected in HF Spaces)"
}

class ConfigManager:
    """Manages API keys and configuration with graceful fallbacks"""
    
    def __init__(self, silent_mode: bool = False):
        self.silent_mode = silent_mode
        self.available_keys = {}
        self.missing_keys = {}
        self.warnings = []
        
        self._check_api_keys()
        
        if not silent_mode:
            self._display_status()
    
    def _check_api_keys(self):
        """Check which API keys are available"""
        for key, info in API_KEYS_INFO.items():
            value = os.getenv(key)
            if value:
                self.available_keys[key] = value
            else:
                self.missing_keys[key] = info
    
    def _display_status(self):
        """Display API key status to user"""
        if self.available_keys:
            print("✅ Available API Keys:")
            for key in self.available_keys:
                masked_key = f"...{self.available_keys[key][-4:]}" if len(self.available_keys[key]) >= 4 else "***"
                print(f"   {key}: {masked_key}")
        
        if self.missing_keys:
            print("\n⚠️  Missing API Keys:")
            for key, info in self.missing_keys.items():
                print(f"   {key}: {info['purpose']}")
                print(f"      Fallback: {info['fallback']}")
                print(f"      Get key: {info['how_to_get']}\n")
            
            print("💡 To set up API keys, add them to your environment:")
            print("   export GOOGLE_API_KEY='your_key_here'")
            print("   export GOOGLE_SEARCH_API_KEY='your_key_here'")
            print("   # etc.\n")
            
            print("🚀 The agent will run with available features only.")
            print("   Some advanced capabilities may be limited.\n")
    
    def get_key(self, key_name: str) -> Optional[str]:
        """Get an API key with graceful handling"""
        return self.available_keys.get(key_name)
    
    def has_key(self, key_name: str) -> bool:
        """Check if a key is available"""
        return key_name in self.available_keys
    
    def require_key(self, key_name: str, feature_name: str = "this feature") -> str:
        """Require a key or raise informative error"""
        if key_name in self.available_keys:
            return self.available_keys[key_name]
        
        info = API_KEYS_INFO.get(key_name, {})
        error_msg = f"""
❌ Missing API Key: {key_name}

{feature_name} requires the {key_name} environment variable.

Purpose: {info.get('purpose', 'API access')}
Get key: {info.get('how_to_get', 'Check API provider documentation')}

To fix this:
1. Get your API key from the provider
2. Set environment variable: export {key_name}='your_key_here'
3. Restart the application

Fallback: {info.get('fallback', 'Feature will be disabled')}
"""
        raise ValueError(error_msg)
    
    def get_available_tools(self) -> List[str]:
        """Get list of tools that can work with current API keys"""
        available_tools = [
            "MathSolver",  # No API key needed
            "TextPreprocesser",  # No API key needed
            "WikipediaTitleFinder",  # No API key needed
            "WikipediaContentFetcher",  # No API key needed
            "RiddleSolver",  # No API key needed
            "WebPageFetcher"  # No API key needed
        ]
        
        if self.has_key("GOOGLE_SEARCH_API_KEY") and self.has_key("GOOGLE_SEARCH_ENGINE_ID"):
            available_tools.append("GoogleSearchTool")
        else:
            available_tools.append("DuckDuckGoSearchTool")  # Free fallback
        
        if self.has_key("GOOGLE_API_KEY"):
            available_tools.extend([
                "FileAttachmentQueryTool",
                "GeminiVideoQA"
            ])
        
        return available_tools

# Global configuration instance
config = ConfigManager()

def safe_getenv(key: str, default: str = None, feature_name: str = None) -> Optional[str]:
    """Safely get environment variable with user-friendly error"""
    value = os.getenv(key, default)
    
    if value is None and feature_name:
        print(f"⚠️  {key} not set - {feature_name} will use fallback method")
    
    return value

def check_required_keys_interactive() -> bool:
    """Interactive check for required keys"""
    missing = []
    for key, info in API_KEYS_INFO.items():
        if not os.getenv(key):
            missing.append((key, info))
    
    if not missing:
        return True
    
    print("\n" + "="*60)
    print("🔧 GAIA SOLVER AGENT - API KEY SETUP")
    print("="*60)
    print("Some API keys are missing. The agent can still run with limited functionality.\n")
    
    for key, info in missing:
        print(f"❌ {key}")
        print(f"   Purpose: {info['purpose']}")
        print(f"   Fallback: {info['fallback']}")
        print(f"   Get key: {info['how_to_get']}\n")
    
    print("Options:")
    print("1. Continue with limited functionality (recommended for testing)")
    print("2. Exit and set up API keys for full functionality")
    print("3. Show detailed setup instructions")
    
    while True:
        choice = input("\nChoose option (1/2/3): ").strip()
        
        if choice == "1":
            print("✅ Continuing with available features...")
            return True
        elif choice == "2":
            print("Please set up your API keys and restart the agent.")
            return False
        elif choice == "3":
            show_setup_instructions()
        else:
            print("Please enter 1, 2, or 3")

def show_setup_instructions():
    """Show detailed API key setup instructions"""
    print("\n" + "="*60)
    print("🔧 DETAILED API KEY SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1. GOOGLE/GEMINI API KEY (Recommended):")
    print("   • Go to: https://makersuite.google.com/app/apikey")
    print("   • Sign in with Google account")
    print("   • Click 'Create API Key'")
    print("   • Copy the key and run:")
    print("     export GOOGLE_API_KEY='your_key_here'")
    print("   • For Gemini model access:")
    print("     export GEMINI_API_KEY='your_key_here'  # Can be same key")
    
    print("\n2. GOOGLE CUSTOM SEARCH (Optional but recommended):")
    print("   • Go to: https://developers.google.com/custom-search/v1/introduction")
    print("   • Create a Custom Search Engine at: https://programmablesearchengine.google.com/")
    print("   • Get your Search Engine ID")
    print("   • Get API key from Google Cloud Console")
    print("   • Set environment variables:")
    print("     export GOOGLE_SEARCH_API_KEY='your_search_api_key'")
    print("     export GOOGLE_SEARCH_ENGINE_ID='your_engine_id'")
    
    print("\n3. Environment Variable Setup:")
    print("   • For current session:")
    print("     export KEY_NAME='your_key_value'")
    print("   • For permanent setup (add to ~/.zshrc or ~/.bashrc):")
    print("     echo 'export GOOGLE_API_KEY=\"your_key\"' >> ~/.zshrc")
    print("     source ~/.zshrc")
    
    print("\n4. Hugging Face Space Deployment:")
    print("   • Add keys in Space Settings > Repository secrets")
    print("   • Keys will be automatically available as environment variables")
    
    print("\n💡 TIP: You can start with just GOOGLE_API_KEY for basic functionality!")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Demo the configuration manager
    print("GAIA Solver Agent - Configuration Check")
    print("="*50)
    
    config = ConfigManager()
    
    print(f"\nAvailable tools: {', '.join(config.get_available_tools())}")
    
    if not config.available_keys:
        print("\n💡 Run with API keys for full functionality!")
        check_required_keys_interactive()
