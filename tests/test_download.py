"""Exercises the resuming downloader against a local server that drops connections."""
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from kaggle_run import download  # noqa: E402

DATA = bytes(range(256)) * 40_000  # about 10 MB of non-repeating-by-offset content


def serve(handler_class):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_class)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f'http://127.0.0.1:{server.server_port}/file'


def make_handler(drop_after=None, drops=0, html_page=None, ignore_range=False):
    state = {'requests': 0, 'starts': []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state['requests'] += 1
            if html_page is not None:
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(html_page.encode())
                return
            start = int(self.headers['Range'].split('=')[1].rstrip('-'))
            state['starts'].append(start)
            if ignore_range:
                start = 0
                self.send_response(200)
            else:
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{len(DATA) - 1}/{len(DATA)}')
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(DATA) - start))
            self.end_headers()
            body = DATA[start:]
            if state['requests'] <= drops:
                body = body[:drop_after]  # close early, as Drive did at 90%
            self.wfile.write(body)
            self.close_connection = True

    return Handler, state


def test_resumes_after_repeated_drops_and_keeps_all_data(tmp_path):
    handler, state = make_handler(drop_after=1_500_000, drops=3)
    server, url = serve(handler)
    target = tmp_path / 'archive.tar.gz'
    download(url, target, retry_wait=0)
    server.shutdown()
    assert target.read_bytes() == DATA
    assert state['starts'][0] == 0
    assert state['starts'][1] > 0, 'second attempt must continue, not restart'
    assert state['starts'] == sorted(state['starts'])
    assert not (tmp_path / 'archive.tar.gz.part').exists()


def test_incomplete_download_never_appears_as_finished(tmp_path):
    handler, _ = make_handler(drop_after=0, drops=10_000)
    server, url = serve(handler)
    target = tmp_path / 'archive.tar.gz'
    with pytest.raises(SystemExit, match='no progress'):
        download(url, target, max_stalls=2, retry_wait=0)
    server.shutdown()
    assert not target.exists()


def test_server_ignoring_range_restarts_cleanly(tmp_path):
    handler, _ = make_handler(ignore_range=True)
    server, url = serve(handler)
    target = tmp_path / 'archive.tar.gz'
    (tmp_path / 'archive.tar.gz.part').write_bytes(b'stale partial data')
    download(url, target, retry_wait=0)
    server.shutdown()
    assert target.read_bytes() == DATA


def test_html_error_page_is_reported_not_saved(tmp_path):
    handler, _ = make_handler(html_page='<html><title>Quota exceeded</title></html>')
    server, url = serve(handler)
    target = tmp_path / 'archive.tar.gz'
    with pytest.raises(PermissionError, match='Quota exceeded'):
        download(url, target, retry_wait=0)
    server.shutdown()
    assert not target.exists()
