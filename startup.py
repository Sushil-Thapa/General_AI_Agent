#!/usr/bin/env python3
"""
GAIA Solver Agent Startup Script
Checks configuration and provides setup guidance
"""

import os
import sys

def main():
    print("🚀 GAIA Solver Agent - Startup Check")
    print("="*50)
    
    try:
        from config import config, check_required_keys_interactive
        
        print("✅ Configuration module loaded")
        
        # Show current status
        if config.available_keys:
            print(f"✅ Found {len(config.available_keys)} API keys")
            available_tools = config.get_available_tools()
            print(f"✅ {len(available_tools)} tools available")
        else:
            print("⚠️  No API keys found")
            print("🔧 Agent will run with limited functionality")
            
            # Ask user if they want setup guidance
            response = input("\nWould you like to see API key setup instructions? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                from config import show_setup_instructions
                show_setup_instructions()
        
        print("\n🎯 Ready to start!")
        print("Run: python app.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("⚠️  Some modules may be missing")
        print("Run: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
