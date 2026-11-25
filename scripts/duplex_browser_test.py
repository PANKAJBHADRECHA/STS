"""Exercise real native duplex streaming with Chromium's recorded microphone.

A streaming-path pass does not certify conversation quality or sustained speed.
"""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent.parent


def main():
    from playwright.sync_api import sync_playwright,expect
    fixture=ROOT/'.runtime/conversation-fixtures/browser-microphone.wav'
    errors=[]
    with (ROOT/'.runtime/duplex-server-test.log').open('w') as log:
        process=subprocess.Popen([str(ROOT/'.venv-sts/bin/python'),str(ROOT/'scripts/duplex_server.py')],cwd=ROOT,stdout=log,stderr=log,env=__import__('os').environ|{'VECLIB_MAXIMUM_THREADS':'1'})
        try:
            import urllib.request
            for _ in range(100):
                try:
                    urllib.request.urlopen('http://127.0.0.1:8767',timeout=.5).close();break
                except OSError:
                    if process.poll() is not None:raise RuntimeError('Duplex server exited')
                    time.sleep(.1)
            with sync_playwright() as p:
                browser=p.chromium.launch(args=['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',f'--use-file-for-fake-audio-capture={fixture}','--autoplay-policy=no-user-gesture-required'])
                context=browser.new_context(permissions=['microphone'],viewport={'width':1200,'height':900})
                page=context.new_page();page.on('pageerror',lambda e: (errors.append(str(e)),print('Page error:',str(e),flush=True)))
                page.goto('http://127.0.0.1:8767');page.locator('#start').click()
                for attempt in range(60):
                    current=page.evaluate('duplexMetrics')
                    if current['received']>=4 and current['captureDuringPlayback']>=2:break
                    if attempt%5==0:print(page.locator('#status').inner_text(),current,flush=True)
                    if errors or (attempt>2 and page.locator('#start').is_enabled()):
                        raise RuntimeError('Streaming stopped: '+page.locator('#status').inner_text())
                    page.wait_for_timeout(2000)
                else:raise RuntimeError('Native duplex stream timed out')
                metrics=page.evaluate('duplexMetrics');status=page.locator('#status').inner_text()
                page.screenshot(path=str(ROOT/'reports/duplex-desktop.png'),full_page=True)
                page.locator('#stop').click();expect(page.locator('#start')).to_be_enabled()
                page.wait_for_function('stream === null && context === null')
                page.set_viewport_size({'width':390,'height':844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                page.screenshot(path=str(ROOT/'reports/duplex-mobile.png'),full_page=True)
                assert not errors,errors
                browser.close()
            report={'status':'streaming_path_passed','inference_mocked':False,'physical_microphone_tested':False,
                    'device':'Chromium WAV-backed microphone','metrics':metrics,'observed_status':status,
                    'stop_released_microphone_and_audio_context':True,'mobile_overflow':False,
                    'sustained_real_time_qualified':False,'conversation_quality_qualified':False,'console_errors':errors}
            (ROOT/'reports/duplex-browser.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
