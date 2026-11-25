"""Actual local transport rejection/overload tests, using the native model."""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parent.parent


def main():
    from websockets.sync.client import connect
    from websockets.exceptions import InvalidStatus,ConnectionClosed
    cases=[]
    with (ROOT/'.runtime/duplex-transport-server.log').open('w') as log:
        process=subprocess.Popen([str(ROOT/'.venv-sts/bin/python'),str(ROOT/'scripts/duplex_server.py')],cwd=ROOT,stdout=log,stderr=log,env=os.environ|{'VECLIB_MAXIMUM_THREADS':'1'})
        try:
            for _ in range(100):
                try:urllib.request.urlopen('http://127.0.0.1:8767',timeout=.5).close();break
                except OSError:time.sleep(.1)
            try:
                with connect('ws://127.0.0.1:8768',origin='https://example.invalid'):raise AssertionError('Foreign origin accepted')
            except InvalidStatus as e:
                assert e.response.status_code==403
                cases.append({'name':'foreign_origin','http_status':403,'passed':True})
            def ready(ws):
                for _ in range(3):
                    data=json.loads(ws.recv(timeout=90))
                    if data.get('status')=='ready':return
                raise AssertionError('Engine did not become ready')
            with connect('ws://127.0.0.1:8768',origin='http://127.0.0.1:8767') as ws:
                ready(ws)
                ws.send(b'invalid')
                data=json.loads(ws.recv(timeout=10))
                assert data=={'status':'stopped','reason':'ValueError'},data
                cases.append({'name':'malformed_audio_frame','passed':True})
            time.sleep(.5)
            with connect('ws://127.0.0.1:8768',origin='http://127.0.0.1:8767') as ws:
                ready(ws)
                try:
                    for _ in range(100):ws.send(bytes(7680))
                    while True:ws.recv(timeout=30)
                except ConnectionClosed as e:
                    assert e.rcvd and e.rcvd.code==1013,str(e)
                    cases.append({'name':'queue_overload','close_code':1013,'passed':True})
            report={'status':'passed','native_inference_mocked':False,'cases':cases}
            (ROOT/'reports/duplex-transport.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
