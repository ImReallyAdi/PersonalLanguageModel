#!/usr/bin/env python3
"""
Simple HTTP server to serve the chat interface
"""
import http.server
import socketserver
import os
import time

PORT = 5003

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow API calls
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def start_server():
    try:
        with socketserver.TCPServer(("0.0.0.0", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"Chat interface server running at http://0.0.0.0:{PORT}")
            print(f"Open http://0.0.0.0:{PORT}/quick_chat.html to use the chatbot")
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Port {PORT} is already in use, trying port {PORT+1}")
            PORT = PORT + 1
            start_server()
        else:
            print(f"Error starting server: {e}")

if __name__ == "__main__":
    start_server()