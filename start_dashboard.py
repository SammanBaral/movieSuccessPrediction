#!/usr/bin/env python3
"""
Simple HTTP Server for Movie Success Prediction Dashboard
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def start_server(port=8080, directory=None):
    """Start a simple HTTP server"""
    
    if directory:
        os.chdir(directory)
    
    # Create a custom handler that serves files with proper MIME types
    class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Add CORS headers
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
            
        def guess_type(self, path):
            """Override to set correct MIME types"""
            mimetype = super().guess_type(path)
            if path.endswith('.json'):
                return 'application/json'
            elif path.endswith('.js'):
                return 'application/javascript'
            elif path.endswith('.css'):
                return 'text/css'
            return mimetype
    
    # Try to find an available port
    for attempt_port in range(port, port + 10):
        try:
            with socketserver.TCPServer(("", attempt_port), CustomHTTPRequestHandler) as httpd:
                print(f"🎬 Movie Success Prediction Dashboard")
                print(f"📡 Server starting on port {attempt_port}")
                print(f"🌐 Open your browser and go to: http://localhost:{attempt_port}")
                print(f"📁 Serving files from: {os.getcwd()}")
                print(f"⏹️  Press Ctrl+C to stop the server\n")
                
                # Try to open browser automatically
                try:
                    webbrowser.open(f'http://localhost:{attempt_port}')
                    print("🚀 Browser opened automatically!")
                except:
                    print("💡 Please open your browser manually")
                
                print("\n" + "="*50)
                
                httpd.serve_forever()
                
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"Port {attempt_port} is busy, trying {attempt_port + 1}...")
                continue
            else:
                print(f"Error starting server: {e}")
                sys.exit(1)
    
    print("❌ Could not find an available port")
    sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Start Movie Success Prediction Dashboard')
    parser.add_argument('--port', '-p', type=int, default=8080, 
                        help='Port to serve on (default: 8080)')
    parser.add_argument('--directory', '-d', type=str, default=None,
                        help='Directory to serve from (default: current directory)')
    
    args = parser.parse_args()
    
    # Check if required files exist
    required_files = ['index.html', 'styles.css', 'dashboard.js']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all dashboard files are in the current directory.")
        sys.exit(1)
    
    # Check if data file exists
    if not Path('pre_release_movie_dataset.json').exists():
        print("⚠️  Warning: pre_release_movie_dataset.json not found")
        print("   The dashboard will use sample data for demonstration")
        print()
    
    try:
        start_server(args.port, args.directory)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("👋 Thanks for using the Movie Success Prediction Dashboard!")
