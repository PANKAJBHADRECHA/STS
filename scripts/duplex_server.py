"""Experimental local duplex microphone demo; never advertised as qualified.

One connection, bounded queues/context, loopback origin checks, no audio logging.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
PROMPT=('Generate a dialogue between you (Tom) and another speaker (User) based on the given narrative. Follow the specific behavior instructions for you.\n\n'
        'Narrative:\n- You and User have a friendly conversation about your day. Greet User and listen to what they say.\n\n'
        'Your behaviors:\n- backchannels: 2\n- interruptions: 0\n- starts the dialogue: True\n\n'
        'Ensure that the dialogue reflects the behaviours of you.\n<|SOS|>')


class Engine:
    def __init__(self):
        self.model=None
        self.pool=ThreadPoolExecutor(max_workers=1)

    def start(self):
        import mlx.core as mx
        import rustymimi
        from tokenizers import Tokenizer
        from sts_lab.factor_mlx import load_factor
        mx.set_cache_limit(32*1024**2)
        mx.set_memory_limit(3*1024**3)
        if self.model is None:self.model=load_factor(ROOT/'.runtime/factor-source')
        self.tokenizer=Tokenizer.from_file(str(ROOT/'.runtime/factor-source/tokenizer.json'))
        speaker=json.loads((ROOT/'.runtime/factor-speaker_embeddings.json').read_text())['tom']
        mx.random.seed(42)
        self.model.reset(self.tokenizer.encode(PROMPT).ids,speaker)
        self.codec=rustymimi.Tokenizer(str(ROOT/'.runtime/native/models/tokenizer-e351c8d8-checkpoint125.safetensors'),num_codebooks=8)
        self.frames=0
        # Compile/warm codec and decoder kernels before enabling capture.
        for _ in range(2):self.step(bytes(7680))
        self.model.reset(self.tokenizer.encode(PROMPT).ids,speaker)
        self.codec.reset()
        self.frames=0

    def step(self,raw):
        import numpy as np
        import mlx.core as mx
        incoming=np.frombuffer(raw,dtype='<f4').copy().reshape(1,1,1920)
        if not np.isfinite(incoming).all() or np.max(np.abs(incoming))>1.01:raise ValueError('Invalid input audio')
        before=time.perf_counter()
        encoded=self.pool.submit(self.codec.encode_step,incoming)
        own,text=self.model.predict()
        codes=np.array(own,dtype=np.uint32)
        if np.any(codes>=2048):raise ValueError('Model ended its audio stream')
        user=encoded.result()
        decoded=self.pool.submit(self.codec.decode_step,codes[:,:,None])
        self.model.advance(own,mx.array(user[:,:,0]),text)
        output=np.clip(decoded.result().reshape(-1),-1,1).astype('<f4').tobytes()
        self.frames+=1
        return output,self.tokenizer.decode([int(text.item())]),time.perf_counter()-before


class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        files={'/':'duplex.html','/duplex-worklet.js':'duplex-worklet.js'}
        name=files.get(self.path)
        if not name:self.send_error(404);return
        body=(ROOT/'web'/name).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type','text/html; charset=utf-8' if name.endswith('.html') else 'text/javascript')
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass


async def serve():
    from websockets.asyncio.server import serve as websocket_server
    executor=ThreadPoolExecutor(max_workers=1)
    engine=Engine()
    active=asyncio.Lock()
    loop=asyncio.get_running_loop()

    async def connection(ws):
        if active.locked():
            await ws.close(1013,'One conversation at a time');return
        async with active:
            queue=asyncio.Queue(maxsize=16)
            reader=None
            try:
                await ws.send(json.dumps({'status':'loading'}))
                await loop.run_in_executor(executor,engine.start)
                await ws.send(json.dumps({'status':'ready','sample_rate':24000,'frame_samples':1920}))
                async def receive():
                    try:
                        async for raw in ws:
                            if not isinstance(raw,bytes) or len(raw)!=7680:raise ValueError('Expected an 80 ms float32 audio frame')
                            try:queue.put_nowait(raw)
                            except asyncio.QueueFull:
                                await ws.close(1013,'Processing cannot keep up with live audio');return
                    except Exception as error:
                        if not queue.full():queue.put_nowait(error)
                    finally:
                        if not queue.full():queue.put_nowait(EOFError('Connection closed'))
                reader=asyncio.create_task(receive())
                for index in range(750):
                    raw=await asyncio.wait_for(queue.get(),timeout=5)
                    if isinstance(raw,Exception):raise raw
                    output,text,elapsed=await loop.run_in_executor(executor,engine.step,raw)
                    await ws.send(output)
                    await ws.send(json.dumps({'status':'streaming','frame':index+1,'text':text,'processing_ms':elapsed*1000,'queued_frames':queue.qsize()}))
                    if reader.done():
                        reader.result();break
                else:await ws.close(1000,'60-second experiment complete; start a new conversation')
            except Exception as error:
                # Never include input audio or authentication information in errors.
                try:await ws.send(json.dumps({'status':'stopped','reason':type(error).__name__}))
                except Exception:pass
                await ws.close()
            finally:
                if reader:
                    reader.cancel()
                    await asyncio.gather(reader,return_exceptions=True)

    page=ThreadingHTTPServer(('127.0.0.1',8767),Page)
    threading.Thread(target=page.serve_forever,daemon=True).start()
    print('Experimental duplex: http://127.0.0.1:8767',flush=True)
    try:
        async with websocket_server(connection,'127.0.0.1',8768,origins=['http://127.0.0.1:8767'],max_size=7680,max_queue=4):
            await asyncio.Future()
    finally:
        page.shutdown();page.server_close()
        executor.shutdown(wait=True);engine.pool.shutdown(wait=True)


if __name__=='__main__':asyncio.run(serve())
