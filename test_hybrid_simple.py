# -*- coding: utf-8 -*-
"""
Test Hybrid Retriever - Direct Import Test
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set UTF-8 encoding for stdout
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def main():
    print("=" * 50)
    print("Testing HybridRetrieverExecutor")
    print("=" * 50)
    
    # Check API keys
    print("\n[API Key Status]")
    amap = os.getenv("AMAP_API_KEY", "")
    baidu = os.getenv("BAIDU_AK", "")
    serpapi = os.getenv("SERPAPI_KEY", "")
    print(f"  AMAP_API_KEY: {'OK' if amap else 'MISSING'}")
    print(f"  BAIDU_AK: {'OK' if baidu else 'MISSING'}")
    print(f"  SERPAPI_KEY: {'OK' if serpapi else 'MISSING'}")
    
    try:
        print("\n[Import Test]")
        from geoagent.executors.hybrid_retriever_executor import (
            HybridRetrieverExecutor,
            SearchResult,
            GeocodedPoint,
            WebSearcher,
            POIDataSource,
            Geocoder,
        )
        print("  All imports successful!")
        
        # Create executor
        print("\n[Executor Creation]")
        executor = HybridRetrieverExecutor()
        print(f"  Executor created: {type(executor).__name__}")
        print(f"  Task type: {executor.task_type}")
        print(f"  Supported engines: {executor.supported_engines}")
        
        # Test with simple query (will fail without API key, but tests the flow)
        print("\n[Test Execution]")
        print("  Running with query='test'...")
        result = executor.run({
            "query": "coffee",
            "city": "beijing",
            "do_geocode": True,
            "do_buffer": False,
        })
        
        print(f"  Success: {result.success}")
        print(f"  Engine: {result.engine}")
        print(f"  Error: {result.error or 'None'}")
        
        if result.data:
            print(f"  Total searched: {result.data.get('total_searched', 0)}")
            print(f"  Total geocoded: {result.data.get('total_geocoded', 0)}")
            print(f"  Summary: {result.data.get('summary', 'N/A')}")
        
        # Test via router
        print("\n[Router Test]")
        from geoagent.executors.router import execute_task
        print("  Router imported successfully")
        
        # Test data structures
        print("\n[Data Structures Test]")
        sr = SearchResult(
            name="Test Store",
            address="123 Main St",
            fuzzy_address="Test Store 123 Main St",
            source="test",
            confidence=0.9
        )
        print(f"  SearchResult: {sr.name} ({sr.confidence})")
        
        gp = GeocodedPoint(
            name="Test",
            input_address="123 Main St",
            lon=116.4,
            lat=39.9,
            formatted_address="Beijing",
            provider="test"
        )
        print(f"  GeocodedPoint: {gp.name} at [{gp.lon}, {gp.lat}]")
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        print(f"\n[ERROR] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
