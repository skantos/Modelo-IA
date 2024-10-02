import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
import tensorflow as tf
import tensorflowjs as tfjs

class MyHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')  # Permitir CORS
        SimpleHTTPRequestHandler.end_headers(self)

def run(server_class=HTTPServer, handler_class=MyHandler, port=8000):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Cambia al directorio del archivo app.py
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Serving HTTP on port {port} ...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
